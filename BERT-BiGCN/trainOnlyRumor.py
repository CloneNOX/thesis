import torch
from torch import optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, random_split
import argparse
import numpy as np
from BertBiGCN import BertBiGCNOnlyRumor
from transformers import BertTokenizer, BertModel
from rumorDataset import RumorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
parser.add_argument('--GCN_hidden_dim', type=int, default=768,\
                    help='dimention of GCN hidden layer, default: 768')
parser.add_argument('--rumor_feature_dim', type=int, default=768,\
                    help='dimention of GCN output, default: 768')
parser.add_argument('--dropout', type=float, default=0.1,\
                    help='dropout rate for model, default: 0.1')
# dataset parameters
parser.add_argument('--data_path', type=str, default='../datasets/PHEME/',\
                    help='path to training dataset, default: ../datasets/PHEME/')
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
parser.add_argument('--log_file', type=str, default='../log/log-only-rumor.txt',\
                    help='log file name, default: ../log/log-only-rumor.txt')
parser.add_argument('--savePath', type=str, default='../model/best-only-rumor.pt',\
                    help='path to save model, default: ../model/best-only-rumor.pt')

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
        dataset = RumorDataset(args.data_path, 'all', tokenizer=tokenizer)
        train_size = dataset.__len__() // 10 * 8
        dev_size = dataset.__len__() // 10 * 9 - train_size
        test_size = dataset.__len__() - train_size - dev_size
        train_set, dev_set, test_set = random_split(dataset, [train_size, dev_size, test_size])

        train_loader = DataLoader(train_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
        dev_loader = DataLoader(dev_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
        test_loader = DataLoader(test_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
        category = dataset.category
    else:
        dataset = RumorDataset(args.data_path, 'traindev', tokenizer=tokenizer)
        train_size = dataset.__len__() // 10 * 8
        dev_size = dataset.__len__() - train_size
        train_set, dev_set = random_split(dataset, [train_size, dev_size])
        
        test_set = RumorDataset(args.data_path, 'test', tokenizer=tokenizer)

        train_loader = DataLoader(train_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
        dev_loader = DataLoader(dev_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
        test_loader = DataLoader(test_set, shuffle=True, collate_fn=RumorDataset.collate_fn)
        category = dataset.category
    torch.manual_seed(time())
    print('done.', flush=True)

    # 选定实验设备
    if 'cuda' in args.device:
        device = torch.device((args.device if torch.cuda.is_available() else 'cpu'))
        if torch.cuda.is_available():
            trainInfo = 'train BERT-BiGCN on device:' + args.device + '\n'
        else:
            trainInfo = 'train BERT-BiGCN on device: cpu\n'
    else:
        device = torch.device('cpu')
        trainInfo = 'train BERT-BiGCN on device: cpu\n'
    logging.info(trainInfo)
    
    # 声明模型、损失函数、优化器
    bert_model = BertModel.from_pretrained(args.bert_path)

    model = BertBiGCNOnlyRumor(
        bert_model = bert_model,
        embedding_dim = args.embedding_dim,
        GCN_hidden_dim = args.GCN_hidden_dim, # GCN隐藏层的维度（GCNconv1的输出维度）
        rumor_feature_dim = args.rumor_feature_dim, # GCN输出层的维度
        rumor_tag_num = len(category), # 谣言标签种类数
        batch_first = True,
        dropout = args.dropout, # 模型默认使用的drop out概率
        device = device
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

    trainRumorAcc = []
    trainRumorF1 = []
    trainLoss = []

    devRumorAcc = []
    devRumorF1 = []
    devLoss = []

    # epochTime = []

    for epoch in range(start, args.epoch + 1):
        logging.info('[epoch: {:d}] '.format(epoch))

        # 训练模型
        rumorTruth = []
        rumorPredict = []
        totalLoss = 0.
        
        model.train()
        # startTime = process_time()
        for node_token, edge_index_TD, edge_index_BU, root_index, label in tqdm(
            train_loader, 
            desc="[epoch: {:d}, training] ".format(epoch), 
            leave=False, 
            ncols=80
        ):
            label = label.to(device)
            rumorTruth += label.tolist()

            optimizer.zero_grad()
            result = model.forward(node_token, edge_index_TD, edge_index_BU, root_index)
            loss = loss_func(result, label)
            loss.requires_grad_(True)
            totalLoss += loss
            loss.backward()
            optimizer.step()
            
            result = softmax(result, dim=1)
            rumorPredict += result.max(dim=1)[1].tolist()
        
        logging.info('average loss: {:f}'.format(
            totalLoss / len(train_loader)
        ))
        trainLoss.append((totalLoss / len(train_loader)).item())
        
        # acc = (np.array(rumorTruth) == np.array(rumorPredict)).sum() / len(rumorTruth)
        acc = accuracy_score(rumorTruth, rumorPredict)
        macroF1 = f1_score(rumorTruth, rumorPredict, labels=range(len(category)), average='macro')
        precision = precision_score(rumorTruth, rumorPredict, labels=range(len(category)), average='macro')
        recall = recall_score(rumorTruth, rumorPredict, labels=range(len(category)), average='macro')
        
        logging.info("    rumor detection accuracy: {:.4f}, macro-f1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(
            acc,
            macroF1,
            precision,
            recall
        ))
        trainRumorAcc.append(acc)
        trainRumorF1.append(macroF1)
        maxF1Train = max(maxF1Train, macroF1)
        # 计算训练时间
        # endTime = process_time()
        # epochTime.append(endTime - startTime)
    
        # 验证集测试并保存最好模型
        logging.info('======')
        logging.info('test model on dev set')
        rumorTruth = []
        rumorPredict = []
        totalLoss = 0.

        model.eval()
        for node_token, edge_index_TD, edge_index_BU, root_index, label in tqdm(
            dev_loader, 
            desc="[epoch: {:d}, dev set test]".format(epoch), 
            leave=False, 
            ncols=80
        ):
            with torch.no_grad():
                rumor_label = label.to(device)
                rumorTruth += label.tolist()

                result = model.forward(node_token, edge_index_TD, edge_index_BU, root_index)
                loss = loss_func(result, rumor_label)
                totalLoss += loss

                result = softmax(result, dim=1)
                rumorPredict += result.max(dim=1)[1].tolist()

        # acc = (np.array(rumorTruth) == np.array(rumorPredict)).sum() / len(rumorTruth)
        acc = accuracy_score(rumorTruth, rumorPredict)
        macroF1 = f1_score(rumorTruth, rumorPredict, labels=range(len(category)), average='macro')
        precision = precision_score(rumorTruth, rumorPredict, labels=range(len(category)), average='macro')
        recall = recall_score(rumorTruth, rumorPredict, labels=range(len(category)), average='macro')
            
        #==============================================
        # 保存验证集marco F1和最大时的模型
        if macroF1 > maxF1Dev:
            model.save(args.savePath)
            earlyStopCounter = 0
            saveStatus = {
                'epoch': epoch,
                'macroF1Rumor': macroF1,
                'accRumor': acc,
                'precisionRumor': precision,
                'recallRumor': recall,
                'loss': (totalLoss / len(dev_loader)).item()
            }
            maxF1Dev = max(maxF1Dev, macroF1)
            logging.info('saved model')
        else:
            earlyStopCounter += 1
        #==============================================
        logging.info('average loss: {:f}'.format(totalLoss / len(dev_loader)))
        logging.info('    rumor detection: accuracy: {:f}, macro-f1: {:f}, precision: {:.4f}, recall: {:.4f}'.format(
            acc, 
            macroF1,
            precision,
            recall
        ))
        logging.info('early stop counter: {:d}'.format(earlyStopCounter))
        logging.info('========================================')
        devLoss.append((totalLoss / len(dev_loader)).item())
        devRumorAcc.append(acc)
        devRumorF1.append(macroF1)
        if earlyStopCounter >= args.patience: # 验证集上连续多次测试性能没有提升就早停
            logging.info('early stop when F1 on dev set did not increase')
            break
    logging.info(saveStatus)
    logging.info('\n\nbest model save at epoch {:d},\
            \nmarco F1 rumor: {:f}, acc rumor {:f}\n'.format(
                saveStatus['epoch'],
                saveStatus['macroF1Rumor'], saveStatus['accRumor']
            ))

    # saveStatus['trainRumorAcc'] = trainRumorAcc
    # saveStatus['trainRumorF1'] = trainRumorF1
    # saveStatus['trainLoss'] = trainLoss

    # saveStatus['devRumorAcc'] = devRumorAcc
    # saveStatus['devRumorF1'] = devRumorF1
    # saveStatus['devLoss'] = devLoss
    # saveStatus['runTime'] = epochTime
    # with open(args.logName[:-4] + '.json', 'w') as f:
    #     f.write(json.dumps(saveStatus))

    # 在测试集测试
    rumorTruth = []
    rumorPredict = []
    totalLoss = 0.
    logging.info('test model on test set')
    model.eval()
    for node_token, edge_index_TD, edge_index_BU, root_index, label in tqdm(
        test_loader, 
        desc="[test set test]", 
        leave=False, 
        ncols=80
    ):
        with torch.no_grad():
            rumor_label = label.to(device)
            rumorTruth += label.tolist()

            result = model.forward(node_token, edge_index_TD, edge_index_BU, root_index)
            loss = loss_func(result, rumor_label)
            totalLoss += loss

            result = softmax(result, dim=1)
            rumorPredict += result.max(dim=1)[1].tolist()

    # acc = (np.array(rumorTruth) == np.array(rumorPredict)).sum() / len(rumorTruth)
    accRumor = accuracy_score(rumorTruth, rumorPredict)
    macroF1 = f1_score(rumorTruth, rumorPredict, labels=range(len(category)), average='macro')
    precision = precision_score(rumorTruth, rumorPredict, labels=range(len(category)), average='macro')
    recall = recall_score(rumorTruth, rumorPredict, labels=range(len(category)), average='macro')
    logging.info('average loss: {:f}'.format(totalLoss / len(dev_loader)))
    logging.info('rumor detection: accuracy: {:f}, macro-f1: {:f}, precision: {:.4f}, recall: {:.4f}'.format(
        accRumor, 
        macroF1,
        precision,
        recall
    ))
# end main()

if __name__ == '__main__':
    main()