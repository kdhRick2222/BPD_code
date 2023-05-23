import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F

from KNN_model import *
from data_loader_multi import *
from earlystop import EarlyStopping
from sklearn.preprocessing import label_binarize

# python3 train.py --target 'bpd_multi'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="./knn_preprocessed.xlsx")
    parser.add_argument('--outputs-dir', type=str, default="./weight")
    parser.add_argument('--target', type=str, default="rds_bi_model2")
    parser.add_argument('--refer', type=str, default="supportv1")
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir)
    out_path = args.outputs_dir
    target = args.target
    # refer = args.refer

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print("TARGET IS ", target)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    criterion3 = nn.BCELoss()

    if target == 'bpd_multi_model2':
        model = KNN_bpd_multi_model2().to(device)
        train_dataset = TrainDataset_multi_model2(args.data, args.refer)
        eval_dataset = EvalDataset_multi_model2(args.data, args.refer)

    elif target == 'bpd_bi_model2':
        model = KNN_bpd_bi_model2().to(device)
        train_dataset = TrainDataset_bi_model2(args.data, args.refer)
        eval_dataset = EvalDataset_bi_model2(args.data, args.refer)

    elif target == 'rds_multi_model2':
        model = KNN_bpd_multi_model2().to(device)
        train_dataset = TrainDataset_multi_rds_model2(args.data, args.refer)
        eval_dataset = EvalDataset_multi_rds_model2(args.data, args.refer)

    elif target == 'rds_bi_model2':
        model = KNN_bpd_bi_model2().to(device)
        train_dataset = TrainDataset_bi_rds_model2(args.data, args.refer)
        eval_dataset = EvalDataset_bi_rds_model2(args.data, args.refer)

    else:
        print('There is no target named like ', target)

    learning_rate = args.lr

    # print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=False,
                                  drop_last=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    print('Data is loaded . . .')

    min_loss = 10
    best_auc, best_acc, best_epoch = 0, 0, 0
    
    for epoch in range(args.num_epochs):

        model.train()

        loss_sum = 0
        cnt = 0

        for data in train_dataloader:
            
            inputs, labels__ = data

            refer = labels__[:, 1].unsqueeze(1).to(device)
            labels_ = labels__[:, 0].unsqueeze(1)
            # print(inputs.size(), refer.size(), labels_.size())
            
            inputs = inputs.to(device)

            if 'multi' in target:
                labels = F.one_hot(labels_.to(torch.int64), num_classes=4).to(device) # one hot vector
                labels_ = labels_.to(device) # class id
            else:
                labels_ = labels_.to(device)
            
            preds_refer, preds = model(inputs)

            # print(preds.size(), labels.squeeze(1).size(), labels_.squeeze(1).size())
            # break
            
            if 'multi' in target:
                loss1 = criterion1(preds, labels_.long().squeeze(1))
                loss2 = criterion2(preds, labels.float().squeeze(1))
                if args.refer == 'rds':
                    loss_r = criterion3(preds_refer, refer)
                else:
                    loss_r = criterion2(preds_refer, refer)
                # loss = 0.01*loss1 + loss2 + loss_r
                loss = loss2 + 0.1*loss_r
                # print(loss1, loss2, loss_r)

            else:
                loss1 = criterion3(preds, labels_)
                loss2 = criterion2(preds, labels_)
                if args.refer == 'rds':
                    loss_r = criterion3(preds_refer, refer)
                else:
                    loss_r = criterion2(preds_refer, refer)
                loss = loss1 + loss2 + loss_r
            
            loss_sum += loss
            cnt += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss)

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        # print("LOSS SUM IS ", loss_sum)

        model.eval()

        accuracy = 0
        testloss = 0
        n = 0
        label = []
        predict = []
        n1, n2, a1, a2 = 0, 0, 0, 0

        for data in eval_dataloader:

            inputs, labels_ = data

            refer = labels_[:, 1].unsqueeze(1).to(device)
            labels_ = labels_[:, 0].unsqueeze(1)
            # print(labels_.size())

            inputs = inputs.to(device)

            if 'multi' in target:
                labels = F.one_hot(labels_.to(torch.int64), num_classes=4).to(device) # one hot vector
                labels_ = labels_.to(device) # class id
            else:
                labels_ = labels_.to(device)

            with torch.no_grad():
                preds_refer, preds = model(inputs)
            
            label += labels_.tolist()
            # predict += preds.tolist()

            # if target == 'bpd_multi' or target == 'rds_multi':
            # print("HIHIHIHIHI")
            if 'multi' in target:
                # testloss += criterion(preds, labels_.to(torch.float32).squeeze(1)) + criterion2(preds, labels_.to(torch.float32).squeeze(1))
                testloss += criterion2(preds, labels.float().squeeze(1))
            else:
                testloss += criterion2(preds, labels_)

            
            # if target == 'bpd_multi' or target == 'rds_multi':
            if 'multi' in target:
                preds_ = preds.argmax()
            else:
                preds_ = torch.round(preds)

            # print(labels, preds_)
            if preds_ == labels_:
                accuracy += 1
            else:
                accuracy += 0

            n += 1

            predict += preds.tolist()

        # print(np.array(label).shape, np.array(predict).shape, preds.size())
        if target == 'bpd_multi_model2' or target == 'rds_multi_model2':
            label_ = label_binarize(label, classes=[0, 1, 2, 3])
            auc_acc = roc_auc_score(label_, predict, multi_class='ovo')
            # accur = accuracy_score(label, np.argmax(np.array(predict), 1))
        else:
            # print(np.array(label).shape, np.array(predict).shape, np.array(label).min(), np.array(label).max())
            auc_acc = roc_auc_score(label, predict)
            # accur = accuracy_score(label, predict)

        testloss /= n
        accur = accuracy/n
        # accur = (a1/n1 + a2/n2)*0.5

        # print("Test Loss : {}".format(testloss))
        if best_acc < accur:
            best_acc = accur
        else:
            best_acc = best_acc

        if best_auc < auc_acc:
            best_auc = auc_acc
            best_epoch = epoch
        else:
            best_auc = best_auc
            best_epoch = best_epoch

        print("EPOCH : {0:3d}  AUCacc : {1:0.4f}  Test : {2:0.4f}  BEST_auc : {3:0.4f}  BEST_acc : {4:0.4f}  BEST_epoch : {5:3d}".format(epoch, auc_acc, accur, best_auc, best_acc, best_epoch))

        if epoch > 2:
            early_stopping(testloss, model)
            
            if min_loss > testloss:
                min_loss = testloss

        if early_stopping.early_stop:
            print('stop!')
            print('minimun loss is ', min_loss)
            break

