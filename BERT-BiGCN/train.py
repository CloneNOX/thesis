import torch
from torch import optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, random_split
import argparse
import numpy as np
from BertBiGCN import BertBiGCN
from transformers import BertTokenizer, BertModel
from rumorDataset import RumorStanceDataset
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
parser.add_argument('--AttentionHeads', type=int, default=8,\
                    help='heads of multi-heads self-attention in sentence2vec layer, default: 8')
parser.add_argument('--dropout', type=float, default=0.1,\
                    help='dropout rate for model, default: 0.1')
parser.add_argument('--need_stance', type=int, default='1',\
                    help='whether use stance feature in rumor detection, default: 1')
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
parser.add_argument('--epoch', type=int, default=40,\
                    help='epoch to train, default: 40')
parser.add_argument('--patience', type=int, default=5,\
                    help='epoch to stop training, default: 5')
parser.add_argument('--device', type=str, default='cuda',\
                    help='select device(cuda/cpu), default: cuda')
parser.add_argument('--log_file', type=str, default='../log/log.txt',\
                    help='log file name, default: ../log/log.txt')
parser.add_argument('--savePath', type=str, default='../model/best.pt',\
                    help='path to save model, default: ../model/best.pt')
parser.add_argument('--acceptThreshold', type=float, default=0.05,\
                    help='threshold for accept model in multi task, default: 0.05')

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
        rumor_category = dataset.rumor_category
        stance_category = dataset.stance_category
    else:
        dataset = RumorStanceDataset(args.data_path, 'traindev', tokenizer=tokenizer)
        train_size = dataset.__len__() // 10 * 8
        dev_size = dataset.__len__() - train_size
        train_set, dev_set = random_split(dataset, [train_size, dev_size])
        
        test_set = RumorStanceDataset(args.data_path, 'test', tokenizer=tokenizer)

        train_loader = DataLoader(train_set, shuffle=True, collate_fn=RumorStanceDataset.collate_fn)
        dev_loader = DataLoader(dev_set, shuffle=True, collate_fn=RumorStanceDataset.collate_fn)
        test_loader = DataLoader(test_set, shuffle=True, collate_fn=RumorStanceDataset.collate_fn)
        rumor_category = dataset.rumor_category
        stance_category = dataset.stance_category
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

    model = BertBiGCN(
        bert_model = bert_model,
        embedding_dim = args.embedding_dim,
        GCN_hidden_dim = args.GCN_hidden_dim, # GCN隐藏层的维度（GCNconv1的输出维度）
        rumor_feature_dim = args.rumor_feature_dim, # GCN输出层的维度
        rumor_label_num = len(rumor_category),
        stance_label_num = len(stance_category),
        need_stance = bool(args.need_stance)
    )
    model = model.set_device(device)
    logging.info(model.__str__())
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

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
    maxRumorF1Train = 0.
    maxStanceF1Train = 0.
    savedRumorF1Dev = 0.
    savedStanceF1Dev = 0.

    trainRumorAcc = []
    trainRumorF1 = []
    trainStanceAcc = []
    trainStanceF1 = []
    trainLoss = []

    devRumorAcc = []
    devRumorF1 = []
    devStanceAcc = []
    devStanceF1 = []
    devLoss = []

    # epochTime = []

    for epoch in range(start, args.epoch + 1):
        logging.info('[epoch: {:d}] '.format(epoch))

        # 训练模型
        rumorTruth = []
        rumorPredict = []
        stanceTruth = []
        stancePredict = []
        totalLoss = 0.
        
        model.train()
        # startTime = process_time()
        for node_token, edge_index_TD, edge_index_BU, root_index, rumor_label, stance_label in tqdm(
            iter(train_loader), 
            desc="[epoch: {:d}] ".format(epoch), 
            leave=False, 
            ncols=80
        ):
            rumor_label = rumor_label.to(device)
            rumorTruth += rumor_label.tolist()
            stance_label = stance_label.to(device)
            stanceTruth += stance_label.tolist()

            optimizer.zero_grad()
            rumorResult, stanceResult = model.forward(node_token, edge_index_TD, edge_index_BU, root_index)
            loss = loss_func(rumorResult, rumor_label) + loss_func(stanceResult, stance_label)
            loss.requires_grad_(True)
            totalLoss += loss
            loss.backward()
            optimizer.step()
            
            rumorResult = softmax(rumorResult, dim=1)
            rumorPredict += rumorResult.max(dim=1)[1].tolist()
            stanceResult = softmax(stanceResult, dim=1)
            stancePredict += stanceResult.max(dim=1)[1].tolist()

        logging.info('average joint-loss: {:f}'.format(
            totalLoss / len(train_loader)
        ))
        trainLoss.append((totalLoss / len(train_loader)).item())

        # acc = (np.array(rumorTruth) == np.array(rumorPredict)).sum() / len(rumorTruth)
        acc = accuracy_score(rumorTruth, rumorPredict)
        macroF1 = f1_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')
        precision = precision_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')
        recall = recall_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')
        
        logging.info("    rumor detection accuracy: {:.4f}, macro-f1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(
            acc,
            macroF1,
            precision,
            recall
        ))
        trainRumorAcc.append(acc)
        trainRumorF1.append(macroF1)

        # acc = (np.array(stanceTruth) == np.array(stancePredict)).sum() / len(stanceTruth)
        acc = accuracy_score(stanceTruth, stancePredict)
        macroF1 = f1_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
        precision = precision_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
        recall = recall_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
        
        logging.info("    stance classification accuracy: {:.4f}, macro-f1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(
            acc,
            macroF1,
            precision,
            recall
        ))
        trainStanceAcc.append(acc)
        trainStanceF1.append(macroF1)
        # 计算训练时间
        # endTime = process_time()
        # epochTime.append(endTime - startTime)
    
        # 测试并保存最好模型
        logging.info('======')
        logging.info('test model on dev set')
        rumorTruth = []
        rumorPredict = []
        stanceTruth = []
        stancePredict = []
        totalLoss = 0.

        model.eval()
        for node_token, edge_index_TD, edge_index_BU, root_index, rumor_label, stance_label in tqdm(
            iter(dev_loader), 
            desc="[epoch: {:d}] ".format(epoch), 
            leave=False, 
            ncols=80
        ):
            with torch.no_grad():
                rumor_label = rumor_label.to(device)
                rumorTruth += rumor_label.tolist()
                stance_label = stance_label.to(device)
                stanceTruth += stance_label.tolist()
                
                rumorResult, stanceResult = model.forward(node_token, edge_index_TD, edge_index_BU, root_index)
                loss = loss_func(rumorResult, rumor_label) + loss_func(stanceResult, stance_label)
                totalLoss += loss

                rumorResult = softmax(rumorResult, dim=1)
                rumorPredict += rumorResult.max(dim=1)[1].tolist()
                stanceResult = softmax(stanceResult, dim=1)
                stancePredict += stanceResult.max(dim=1)[1].tolist()
        
        accRumor = accuracy_score(rumorTruth, rumorPredict)
        macroF1Rumor = f1_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')
        precisionRumor = precision_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')
        recallRumor = recall_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')

        accStance = accuracy_score(stanceTruth, stancePredict)
        macroF1Stance = f1_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
        precicionStance = precision_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
        recallStance = recall_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
            
        #==============================================
        # 保存验证集marco F1和最大时的模型
        if accept(
            newRumorF1 = macroF1Rumor,
            savedRumorF1 = savedRumorF1Dev,
            newStanceF1 = macroF1Stance,
            savedStanceF1 = savedStanceF1Dev,
            threshold = args.acceptThreshold
        ):
            model.save(args.savePath)
            earlyStopCounter = 0
            saveStatus = {
                'epoch': epoch,
                'macroF1Rumor': macroF1Rumor,
                'macroF1Stance': macroF1Stance,
                'accRumor': accRumor,
                'accStance': accStance,
                'precisionRumor': precisionRumor,
                'precisionStance': precicionStance,
                'recallRumor': recallRumor,
                'recallRumor': recallStance,
                'loss': (totalLoss / len(dev_loader)).item()
            }
            savedRumorF1Dev = macroF1Rumor
            savedStanceF1Dev = macroF1Stance
            logging.info('saved model')
        else:
            earlyStopCounter += 1
        #==============================================
        logging.info('average joint-loss: {:f}'.format(totalLoss / len(dev_loader)))
        logging.info('rumor detection:')
        logging.info('accuracy: {:f}, macro-f1: {:f}, precision: {:.4f}, recall: {:.4f}'.format(
            accRumor, 
            macroF1Rumor,
            precisionRumor,
            recallRumor
        ))
        logging.info('stance analyze:')
        logging.info('accuracy: {:f}, macro-f1: {:f}, precision: {:.4f}, recall: {:.4f}'.format(
            accStance,
            macroF1Stance,
            precicionStance,
            recallStance
        ))
        logging.info('early stop counter: {:d}'.format(earlyStopCounter))
        logging.info('========================================')
        devLoss.append((totalLoss / len(dev_loader)).item())
        devRumorAcc.append(accRumor)
        devRumorF1.append(macroF1Rumor)
        devStanceAcc.append(accStance)
        devStanceF1.append(macroF1Stance)
        if earlyStopCounter >= args.patience: # 验证集上连续多次测试性能没有提升就早停
            logging.info('early stop when F1 on dev set did not increase')
            break
    logging.info(saveStatus)
    logging.info('\n\nsave model at epoch {:d},\
            \nmarco F1 rumor: {:f}, acc rumor {:f}\
            \nmarco F1 stance: {:f}, acc stance: {:f}\n'.format(
                saveStatus['epoch'],
                saveStatus['macroF1Rumor'], saveStatus['accRumor'],
                saveStatus['macroF1Stance'], saveStatus['accStance']
            ))

    # saveStatus['trainRumorAcc'] = trainRumorAcc
    # saveStatus['trainRumorF1'] = trainRumorF1
    # saveStatus['trainStanceAcc'] = trainStanceAcc
    # saveStatus['trainStanceF1'] = trainStanceF1
    # saveStatus['trainLoss'] = trainLoss

    # saveStatus['testRumorAcc'] = testRumorAcc
    # saveStatus['testRumorF1'] = testRumorF1
    # saveStatus['testStanceAcc'] = testStanceAcc
    # saveStatus['testStanceF1'] = testStanceF1
    # saveStatus['testLoss'] = testLoss
    # saveStatus['runTime'] = epochTime
    # with open(args.logName[:-4] + '.json', 'w') as f:
    #     f.write(json.dumps(saveStatus))

    # 在测试集测试
    rumorTruth = []
    rumorPredict = []
    stanceTruth = []
    stancePredict = []
    totalLoss = 0.
    logging.info('test model on test set')
    model.eval()
    for node_token, edge_index_TD, edge_index_BU, root_index, rumor_label, stance_label in tqdm(
        test_loader, 
        desc="[test set test]", 
        leave=False, 
        ncols=80
    ):
        with torch.no_grad():
            rumor_label = rumor_label.to(device)
            rumorTruth += rumor_label.tolist()
            stance_label = stance_label.to(device)
            stanceTruth += stance_label.tolist()
            
            rumorResult, stanceResult = model.forward(node_token, edge_index_TD, edge_index_BU, root_index)
            loss = loss_func(rumorResult, rumor_label) + loss_func(stanceResult, stance_label)
            totalLoss += loss

            rumorResult = softmax(rumorResult, dim=1)
            rumorPredict += rumorResult.max(dim=1)[1].tolist()
            stanceResult = softmax(stanceResult, dim=1)
            stancePredict += stanceResult.max(dim=1)[1].tolist()
    # macroF1Rumor = f1_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')
    # accRumor = (np.array(rumorTruth) == np.array(rumorPredict)).sum() / len(rumorTruth)
    # macroF1Stance = f1_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
    # accStance = (np.array(stanceTruth) == np.array(stancePredict)).sum() / len(stanceTruth)
    accRumor = accuracy_score(rumorTruth, rumorPredict)
    macroF1Rumor = f1_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')
    precisionRumor = precision_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')
    recallRumor = recall_score(rumorTruth, rumorPredict, labels=range(len(rumor_category)), average='macro')

    accStance = accuracy_score(stanceTruth, stancePredict)
    macroF1Stance = f1_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
    precicionStance = precision_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
    recallStance = recall_score(stanceTruth, stancePredict, labels=range(len(stance_category)), average='macro')
    logging.info('average joint-loss: {:f}'.format(totalLoss / len(dev_loader)))
    logging.info('rumor detection:')
    logging.info('accuracy: {:f}, macro-f1: {:f}, precision: {:.4f}, recall: {:.4f}'.format(
        accRumor, 
        macroF1Rumor,
        precisionRumor,
        recallRumor
    ))
    logging.info('stance analyze:')
    logging.info('accuracy: {:f}, macro-f1: {:f}, precision: {:.4f}, recall: {:.4f}'.format(
        accStance,
        macroF1Stance,
        precicionStance,
        recallStance
    ))

# end main()

# 一个辅助接受保存模型的函数
def accept(newRumorF1, savedRumorF1, newStanceF1, savedStanceF1, threshold):
    if newRumorF1 - savedRumorF1 >= 0 and newStanceF1 - savedStanceF1 >= 0:
        return True
    elif newRumorF1 - savedRumorF1 < 0 and newStanceF1 - savedStanceF1 < 0:
        return False
    elif newRumorF1 - savedRumorF1 < 0 and newStanceF1 - savedStanceF1 > 0:
        if abs(newRumorF1 - savedRumorF1) <= threshold * savedRumorF1:
            return True
        else:
            return False
    elif newRumorF1 - savedRumorF1 > 0 and newStanceF1 - savedStanceF1 < 0:
        if abs(newStanceF1 - savedStanceF1) <= threshold * savedRumorF1:
            return True
        else:
            return False
    else:
        return False

if __name__ == '__main__':
    main()