import torch
from torch import optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, random_split
import argparse
import numpy as np
from BertBiGCN import BertMSA
from transformers import BertTokenizer, BertModel
from rumorDataset import RumorStanceDataset
from sklearn.metrics import f1_score
from tqdm import tqdm
from copy import copy
from time import process_time, time
import json
import logging

parser = argparse.ArgumentParser()

# model hyperparameters
parser.add_argument('--bert_path' ,type=str, default='../model/bert-base-cased',\
                    help="path to bert model, default: '../model/bert-base-cased'")
parser.add_argument('--embedding_dim' ,type=int, default=768,\
                    help='dimention of embedding')
parser.add_argument('--AttentionHeads', type=int, default=8,\
                    help='heads of multi-heads self-attention in sentence2vec layer, default: 8')
parser.add_argument('--dropout', type=float, default=0.1,\
                    help='dropout rate for model, default: 0.1')
# dataset parameters
parser.add_argument('--data_path', type=str, default='../datasets/semeval2017-task8/',\
                    help='path to training dataset, default: ../datasets/semeval2017-task8/')
parser.add_argument('--dataset_random_seed', type=int, default=925,\
                    help='random seed for dataset split, fix dataset partition for each train, default: 925')
# train parameters
parser.add_argument('--optimizer', type=str, default='AdamW',\
                    help='set optimizer type in [SGD/Adam/AdamW...], default: AdamW')
parser.add_argument('--lr', type=float, default=1e-5,\
                    help='set learning rate, default: 1e-5')
parser.add_argument('--weightDecay', type=float, default=5e-4,\
                    help='set weight decay for L2 Regularization, default: 5e-4')
parser.add_argument('--epoch', type=int, default=20,\
                    help='epoch to train, default: 20')
parser.add_argument('--patience', type=int, default=5,\
                    help='epoch to stop training, default: 5')
parser.add_argument('--device', type=str, default='cuda',\
                    help='select device(cuda/cpu), default: cuda')
parser.add_argument('--log_file', type=str, default='../log/log-only-stance.txt',\
                    help='log file name, default: ../log/log-only-stance.txt')
parser.add_argument('--savePath', type=str, default='../model/best-only-stance.pt',\
                    help='path to save model, default: ../model/best-only-stance.pt')

def main():
    args = parser.parse_args()
    logging.basicConfig(
        filename=args.log_file, 
        filemode="w", 
        format="[%(asctime)s]:%(levelname)s: %(message)s", 
        datefmt="%d-%M-%Y %H:%M:%S", 
        level=logging.DEBUG
    )
    
    # 获取数据集及词嵌入
    print('preparing data...', end='', flush=True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    torch.manual_seed(args.dataset_random_seed)
    if 'PHEME' in args.data_path:
        dataset = RumorStanceDataset(args.data_path, 'all', tokenizer=tokenizer)
        train_size = dataset.__len__() // 10 * 8
        dev_size = dataset.__len__() // 10 * 9 - train_size
        test_size = dataset.__len__() - train_size - dev_size
        train_set, dev_set, test_set = random_split(dataset, [train_size, dev_size, test_size])

        train_loader = DataLoader(train_set, shuffle=True, collate_fn=RumorStanceDataset.collate_fn)
        dev_loader = DataLoader(dev_set, shuffle=True, collate_fn=RumorStanceDataset.collate_fn)
        test_loader = DataLoader(test_set, shuffle=True, collate_fn=RumorStanceDataset.collate_fn)
        category = dataset.stance_category
    else:
        dataset = RumorStanceDataset(args.data_path, 'traindev', tokenizer=tokenizer)
        train_size = dataset.__len__() // 10 * 8
        dev_size = dataset.__len__() - train_size
        train_set, dev_set = random_split(dataset, [train_size, dev_size])
        
        test_set = RumorStanceDataset(args.data_path, 'test', tokenizer=tokenizer)

        train_loader = DataLoader(train_set, shuffle=True, collate_fn=RumorStanceDataset.collate_fn)
        dev_loader = DataLoader(dev_set, shuffle=True, collate_fn=RumorStanceDataset.collate_fn)
        test_loader = DataLoader(test_set, shuffle=True, collate_fn=RumorStanceDataset.collate_fn)
        category = dataset.stance_category
    torch.manual_seed(time())
    print('done.', flush=True)

    # 选定实验设备
    if 'cuda' in args.device:
        device = torch.device((args.device if torch.cuda.is_available() else 'cpu'))
        if torch.cuda.is_available():
            trainInfo = 'train MSABiGCN on device:' + args.device + '\n'
        else:
            trainInfo = 'train MSABiGCN on device: cpu\n'
    else:
        device = torch.device('cpu')
        trainInfo = 'train MSABiGCN on device: cpu\n'
    logging.info(trainInfo)
    
    # 声明模型、损失函数、优化器
    bert_model = BertModel.from_pretrained(args.bert_path)

    model = BertMSA(
        bert_model = bert_model,
        embedding_dim = args.embedding_dim,
        num_stance_tag = len(category),
        batch_first = True
    )
    model = model.set_device(device)
    logging.info(model.__str__())
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    # 选定需要训练的参数
    bert_params_id = list(map(id, model.bert_model.parameters()))
    bert_params = filter(lambda p:id(p) in bert_params, model.parameters())
    base_params = filter(lambda p:id(p) not in bert_params, model.parameters())
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW([
            {'params': bert_params},
            {'params': base_params}
        ], lr = args.lr, weight_decay = args.weightDecay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': bert_params},
            {'params': base_params}
        ], lr = args.lr, weight_decay = args.weightDecay)
    else:
        optimizer = optim.SGD([
            {'params': bert_params},
            {'params': base_params}
        ], lr = args.lr, momentum = 0.9, weight_decay = args.weightDecay)

    start = 1
    # 记录验证集上的最好性能，用于early stop
    earlyStopCounter = 0
    maxF1Train = 0.
    maxF1Dev = 0.

    trainStanceAcc = []
    trainStanceF1 = []
    trainLoss = []

    devStanceAcc = []
    devStanceF1 = []
    devLoss = []

    for epoch in range(start, args.epoch + 1):
        logging.info('[epoch: {:d}] '.format(epoch))

        # 训练模型
        stanceTruth = []
        stancePredict = []
        totalLoss = 0.
        
        model.train()
        for node_token, edge_index_TD, edge_index_BU, root_index, rumor_label, stance_label in tqdm(
            iter(train_loader), 
            desc="[epoch: {:d}] ".format(epoch), 
            leave=False, 
            ncols=80
        ):
            stance_label = stance_label.to(device)
            stanceTruth += stance_label.tolist()

            optimizer.zero_grad()
            result = model.forward(node_token)
            loss = loss_func(result, stance_label)
            loss.requires_grad_(True)
            totalLoss += loss
            loss.backward()
            optimizer.step()
                
            result = softmax(result, dim=1)
            stancePredict += result.max(dim=1)[1].tolist()
        logging.info('average loss: {:f}'.format(
            totalLoss / len(train_loader)
        ))
        macroF1 = f1_score(stanceTruth, stancePredict, labels=range(len(category)), average='macro')
        acc = (np.array(stanceTruth) == np.array(stancePredict)).sum() / len(stanceTruth)
        logging.info("    stance classification accuracy: {:.4f}, macro-f1: {:.4f}".format(
            acc,
            macroF1
        ))
        trainStanceAcc.append(acc)
        trainStanceF1.append(macroF1)
        maxF1Train = max(maxF1Train, macroF1)
    
        # 验证集测试并保存最好模型
        logging.info('======')
        logging.info('test model on dev set')
        stanceTruth = []
        stancePredict = []
        totalLoss = 0.

        model.eval()
        for node_token, edge_index_TD, edge_index_BU, root_index, rumor_label, stance_label in tqdm(
            dev_loader, 
            desc="[epoch: {:d}, dev set test]".format(epoch), 
            leave=False, 
            ncols=80
        ):
            with torch.no_grad():
                stance_label = stance_label.to(device)
                stanceTruth += stance_label.tolist()

                result = model.forward(node_token)
                loss = loss_func(result, stance_label)
                totalLoss += loss

                result = softmax(result, dim=1)
                stancePredict += result.max(dim=1)[1].tolist()
        
        macroF1 = f1_score(stanceTruth, stancePredict, labels=range(len(category)), average='macro')
        acc = (np.array(stanceTruth) == np.array(stancePredict)).sum() / len(stancePredict)
            
        ##==============================================
        # 保存验证集marco F1和最大时的模型
        if macroF1 > maxF1Dev:
            model.save(args.savePath)
            earlyStopCounter = 0
            saveStatus = {
                'epoch': epoch,
                'macroF1Stance': macroF1,
                'accStance': acc,
                'loss': (totalLoss / len(dev_loader)).item()
            }
            maxF1Dev = max(maxF1Dev, macroF1)
            logging.info('saved model')
        else:
            earlyStopCounter += 1
        #==============================================
        logging.info('average loss: {:f}'.format(totalLoss / len(dev_loader)))
        logging.info('    stance classification accuracy: {:f}, macro-f1: {:f}'.format(
            acc, 
            macroF1
        ))
        logging.info('early stop counter: {:d}'.format(earlyStopCounter))
        logging.info('========================================')
        devLoss.append((totalLoss / len(dev_loader)).item())
        devStanceAcc.append(acc)
        devStanceF1.append(macroF1)
        if earlyStopCounter >= args.patience: # 验证集上连续多次测试性能没有提升就早停
            logging.info('early stop when F1 on dev set did not increase')
            break
    logging.info(saveStatus)
    logging.info(saveStatus)
    logging.info('\n\nbest model save at epoch {:d},\
            \nmarco F1 stance: {:f}, acc stance {:f}\n'.format(
                saveStatus['epoch'],
                saveStatus['macroF1Stance'], saveStatus['accStance']
            ))

    # saveStatus['trainStanceAcc'] = trainStanceAcc
    # saveStatus['trainStanceF1'] = trainStanceF1
    # saveStatus['trainLoss'] = trainLoss

    # saveStatus['testStanceAcc'] = devStanceAcc
    # saveStatus['testStanceF1'] = devStanceF1
    # saveStatus['testLoss'] = devLoss
    # with open(args.logName[:-4] + '.json', 'w') as f:
    #     f.write(json.dumps(saveStatus))
    # 在测试集测试
    
    logging.info('test model on test set')
    stanceTruth = []
    stancePredict = []
    totalLoss = 0.
    model.eval()
    for node_token, edge_index_TD, edge_index_BU, root_index, rumor_label, stance_label in tqdm(
        test_loader, 
        desc="[test set test]", 
        leave=False, 
        ncols=80
    ):
        with torch.no_grad():
            stance_label = stance_label.to(device)
            stanceTruth += stance_label.tolist()

            result = model.forward(node_token)
            loss = loss_func(result, stance_label)
            totalLoss += loss
            
            result = softmax(result, dim=1)
            stancePredict += result.max(dim=1)[1].tolist()

    macroF1 = f1_score(stanceTruth, stancePredict, labels=range(len(category)), average='macro')
    accStance= (np.array(stanceTruth) == np.array(stancePredict)).sum() / len(stancePredict)
    logging.info('average loss: {:f}'.format(totalLoss / len(dev_loader)))
    logging.info('stance detection: accuracy: {:f}, macro-f1: {:f}'.format(
        accStance, 
        macroF1
    ))
# end main()

if __name__ == '__main__':
    main()