import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Binarizer
from numpy import argmax

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import label_binarize

from KNN_model import *
from data_loader_total import *
from data_loader_classicalmethods import *


def roc_curve_plot(l, p1, p2, target):
    fprs1, tprs1, thresholds1 = roc_curve(l, p1)
    fprs2, tprs2, thresholds2 = roc_curve(l, p2)

    plt.clf()

    plt.plot(fprs1, tprs1, label='Model1')
    plt.plot(fprs2, tprs2, label='Model2')
    plt.plot([0,1], [0,1], 'k--', label='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.xlabel('FPR( 1 - Specificity  )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig('/home/daehyun/mnt/nas12/KNN_bpd/roc_curve/'+ target +'.png')


def roc_curve_plot_all(l, l_np, prediction, names, target):
    plt.clf()
    print(len(prediction), len(names))
    for i in range(len(prediction)):
        if (names[i] == 'PMbpd') or (names[i] == 'TS-PMbpd'):
            fprs, tprs, thresholds = roc_curve(l, prediction[i])
        else:
            fprs, tprs, thresholds = roc_curve(l_np, prediction[i])
        plt.plot(fprs, tprs, label=names[i])
    
    plt.plot([0,1], [0,1], 'k--', label='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.xlabel('FPR( 1 - Specificity  )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig('/home/daehyun/mnt/nas12/KNN_bpd/roc_curve/'+ target +'.png')


def roc_curve_plot_multi(l, p1, p2, target):
    for i in range(l.shape[-1]):
        fprs, tprs, _ = roc_curve(l[:,i], p1[:,i]) #calculate fprs and tprs for each class
        # label_name = 'BPD {}'.format(i)
        # print(label_name)
        plt.plot(fprs,tprs,label='BPD {}'.format(i)) #plot roc curve of each class
    plt.xlabel('FPR( 1 - Specificity  )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig('/home/daehyun/mnt/nas12/KNN_bpd/roc_curve/'+ target +'.png')

    plt.clf()
    
    for i in range(l.shape[-1]):
        fprs, tprs, _ = roc_curve(l[:,i], p2[:,i]) #calculate fprs and tprs for each class
        plt.plot(fprs,tprs,label='BPD {}'.format(i)) #plot roc curve of each class
    plt.xlabel('FPR( 1 - Specificity  )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig('/home/daehyun/mnt/nas12/KNN_bpd/roc_curve/'+ target +'_model2.png')


def get_clf_eval(y_test, pred):
       ppv = precision_score(y_test, pred, average='macro')
       sensitivity = recall_score(y_test, pred, average='macro')
       return ppv, sensitivity

def get_clf_eval_for_multi(y_test, pred):
       ppv = precision_score(y_test, pred, average='weighted')
       sensitivity = recall_score(y_test, pred, average='weighted')
       return ppv, sensitivity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="bpd_multi")
    parser.add_argument('--path', type=str, default="/home/daehyun/mnt/nas12/KNN_bpd/knn_preprocessed.xlsx")
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    path = args.path
    target = args.target

    if 'bpd_multi' in target:
        model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/bpd_0.pth'
        model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_bpd_0.pth'
    elif 'bpd_bi1' in target:
        model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/bpd_1.pth'
        model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_bpd_1.pth'
    elif 'bpd_bi2' in target:
        model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/bpd_2.pth'
        model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_bpd_2.pth'
    elif 'bpd_bi3' in target:
        model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/bpd_3.pth'
        model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_bpd_3.pth'
#     elif 'rds1_multi' in target:
#         model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/rds1_multi.pth'
#         model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_rds1_multi.pth'
#     elif 'rds2_multi' in target:
#         model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/rds2_multi.pth'
#         model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_rds2_multi.pth'
#     elif 'rds1_bi1' in target:
#         model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/rds1_bi1.pth'
#         model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_rds1_bi1.pth'
#     elif 'rds1_bi2' in target:
#         model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/rds1_bi2.pth'
#         model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_rds1_bi2.pth'
#     elif 'rds1_bi3' in target:
#         model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/rds1_bi3.pth'
#         model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_rds1_bi3.pth'
#     elif 'rds2_bi1' in target:
#         model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/rds2_bi1.pth'
#         model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_rds2_bi1.pth'
#     elif 'rds2_bi2' in target:
#         model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/rds2_bi2.pth'
#         model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_rds2_bi2.pth'
#     elif 'rds2_bi3' in target:
#         model1_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/rds2_bi3.pth'
#         model2_weight = '/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_rds2_bi3.pth'
    else:
        print("NO TARGET...")

    if 'bi' in target:
        model1 = KNN_bpd_bi().to(device)
        model1.load_state_dict(torch.load(model1_weight))
        model2 = KNN_bpd_bi_model2().to(device)
        model2.load_state_dict(torch.load(model2_weight))
    elif 'multi' in target:
        model1 = KNN_bpd_multi().to(device)
        model1.load_state_dict(torch.load(model1_weight))
        model2 = KNN_bpd_multi_model2().to(device)
        model2.load_state_dict(torch.load(model2_weight))
    
    if target == 'bpd_multi':
        # train = TrainDataset_bpd_multi(path)
        test = EvalDataset_bpd_multi(path)
    elif 'bpd_bi' in target:
        # train = TrainDataset_bpd_bi(path, target)
        test = EvalDataset_bpd_bi(path, target)
    elif (target == 'rds1_multi') or (target == 'rds2_multi'):
        # train = TrainDataset_multi_rds(path, target)
        test = EvalDataset_multi_rds(path, target)
    elif ('rds' in target) and ('bi' in target):
        # train = TrainDataset_bi_rds(path, target)
        test = EvalDataset_bi_rds(path, target)
    else:
        print("There is no Dataloader of the target...")

#######################################################################
    
    if target == 'bpd_multi':
        train_np = TrainDataset_bpd_multi_np(path)
        test_np = EvalDataset_bpd_multi_np(path)
    elif 'bpd_bi' in target:
        train_np = TrainDataset_bpd_bi_np(path, target)
        test_np = EvalDataset_bpd_bi_np(path, target)
    elif (target == 'rds1_multi') or (target == 'rds2_multi'):
        train_np = TrainDataset_multi_rds_np(path, target)
        test_np = EvalDataset_multi_rds_np(path, target)
    elif ('rds' in target) and ('bi' in target):
        train_np = TrainDataset_bi_rds_np(path, target)
        test_np = EvalDataset_bi_rds_np(path, target)
    else:
        print("There is no Dataloader of the target...")


    if 'multi' in target:
        x_test_np = test_np[:][0]
        y_test_np = test_np[:][1]
        x_train_np = train_np[:][0]
        y_train_np = train_np[:][1]
        # print(y_train.shape)
        y_train_np_ = label_binarize(y_train_np, classes=[0, 1, 2, 3])
        y_test_np_ = label_binarize(y_test_np, classes=[0, 1, 2, 3])
    else:
        x_test_np = test_np[:][0]
        y_test_np = test_np[:][1].reshape(-1)
        x_train_np = train_np[:][0]
        y_train_np = train_np[:][1].reshape(-1)


    eval_dataloader = DataLoader(dataset=test, batch_size=1)

    label, predict1, predict2 = [], [], []

    model1.eval()
    model2.eval()

    for data in eval_dataloader:

        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds1 = model1(inputs)
            preds2 = model2(inputs)

        label += labels.tolist()
        predict1 += preds1.tolist()
        predict2 += preds2.tolist()

    # print(np.array(label).shape, np.array(predict1).shape, np.array(predict2).shape)  # (1120, 1), (1120, 4) (1120, 4)
    # print(label)
    values = []

    if 'multi' in target:
        label_np = np.squeeze(np.array(label), axis=1).astype(int)  # (1120,)
        n_values = np.max(label_np).astype(int) + 1
        label_ = np.eye(n_values)[label_np.tolist()]
        # print(label_.shape)
        label_np_ = np.array(label_)
        # print(label_np_.shape)
         
        auroc1 = roc_auc_score(label_, predict1, multi_class='ovo')
        accuracy1 = accuracy_score(label, np.argmax(predict1, 1))
        prediction1 = np.array(predict1)
        ppv1, sensitivity1 = get_clf_eval_for_multi(label, np.argmax(prediction1, 1))
        f11 = f1_score(label, np.argmax(prediction1, 1), average='weighted')
        values.append([ppv1, sensitivity1, f11, auroc1, accuracy1])

        auroc2 = roc_auc_score(label_, predict2, multi_class='ovo')
        accuracy2 = accuracy_score(label, np.argmax(predict2, 1))
        prediction2 = np.array(predict2)
        ppv2, sensitivity2 = get_clf_eval_for_multi(label, np.argmax(prediction2, 1))
        f12 = f1_score(label, np.argmax(prediction2, 1), average='weighted')
        values.append([ppv2, sensitivity2, f12, auroc2, accuracy2])
        
        roc_curve_plot_multi(label_np_, prediction1, prediction2, target)

        models_dataframe=pd.DataFrame(values,index=['model1', 'model2'])
        models_dataframe.columns=['Precision', 'Recall', 'F1-score', 'AUROC', 'Accuracy']


        print(models_dataframe)

    else:
        classifiers=['Linear SVM', 'Radial SVM', 'Logistic Regression', 'KNN', 'Decision Tree', 'XGBOOST', 'LightGBM']
        names = ['Linear SVM', 'Radial SVM', 'Logistic Regression', 'KNN', 'Decision Tree', 'XGBOOST', 'LightGBM', 'PMbpd', 'TS-PMbpd']
        models=[svm.SVC(kernel='linear', probability=True),svm.SVC(kernel='rbf', probability=True),LogisticRegression(),KNeighborsClassifier(),
        DecisionTreeClassifier(),XGBClassifier(use_label_encoder=False, eval_metric='logloss'),LGBMClassifier()]
        prediction_total = []
        
        for i in models:
            model = i
            model.fit(x_train_np, y_train_np.ravel())
            prediction = model.predict_proba(x_test_np)
            length = len(prediction)
            # prediction_sum = np.sum(np.array(prediction), 1).reshape(length, 1)

            # prediction_2 = np.max(np.array(prediction)/np.concatenate((prediction_sum, prediction_sum), 1), 1)
            # prediction = np.argmax(np.array(prediction)/np.concatenate((prediction_sum, prediction_sum), 1), 1)
            predict_ml = np.array(prediction)[:, 1]
            prediction = predict_ml
            # print(y_test_np.shape, predict_ml.shape)
            # print(y_test_np.shape, np.argmax(predict_ml, 1).shape)
            prediction_total.append(predict_ml)

            fpr, tpr, thresholds = roc_curve(y_test_np, prediction)
            J = tpr - fpr
            ix = argmax(J)
            best_threshold = thresholds[ix]
            # predict_ml = np.array(prediction)
            label_np = np.array(y_test_np)
            binarizer = Binarizer(threshold=best_threshold)
            custom_predict = binarizer.fit_transform(predict_ml.reshape(-1, 1))
            ppv, sensitivity = get_clf_eval(label_np, custom_predict)

            f1 = f1_score(label_np, custom_predict, average='weighted')
            auroc = roc_auc_score(y_test_np, custom_predict)
            accuracy = accuracy_score(y_test_np, custom_predict)

            values.append([ppv, sensitivity, f1, auroc, accuracy])


        label_np_ = np.squeeze(np.array(label), axis=1)
        predict1_np = np.squeeze(np.array(predict1), axis=1)
        auroc1 = roc_auc_score(label, predict1)

        fprs1, tprs1, thresholds1 = roc_curve(label, predict1)
        J1 = tprs1 - fprs1
        ix1 = argmax(J1)
        best_threshold1 = thresholds1[ix1]
        binarizer1 = Binarizer(threshold=best_threshold1)
        custom_predict1 = binarizer1.fit_transform(predict1_np.reshape(-1, 1))
        accuracy1 = accuracy_score(label_np_, custom_predict1)
        prediction1 = np.array(predict1)
        label = np.array(label)
        ppv1, sensitivity1 = get_clf_eval(label, custom_predict1)
        f11 = f1_score(label, custom_predict1, average='weighted')
        values.append([ppv1, sensitivity1, f11, auroc1, accuracy1])
        prediction_total.append(prediction1)

        # label_np = np.squeeze(np.array(label), axis=1)
        predict2_np = np.squeeze(np.array(predict2), axis=1)
        auroc2 = roc_auc_score(label, predict2)

        fprs2, tprs2, thresholds2 = roc_curve(label, predict2)
        J2 = tprs2 - fprs2
        ix2 = argmax(J2)
        best_threshold2 = thresholds1[ix2]
        binarizer2 = Binarizer(threshold=best_threshold2)
        custom_predict2 = binarizer2.fit_transform(predict2_np.reshape(-1, 1))
        accuracy2 = accuracy_score(label_np_, custom_predict2)
        prediction2 = np.array(predict2)
        # label = np.array(label)
        ppv2, sensitivity2 = get_clf_eval(label, custom_predict2)
        f12 = f1_score(label, custom_predict2, average='weighted')
        values.append([ppv2, sensitivity2, f12, auroc2, accuracy2])
        prediction_total.append(prediction2)

        # roc_curve_plot(label, predict1, predict2, target)
        roc_curve_plot_all(label, label_np, prediction_total, names, target)


    # roc_curve_plot(label, predict1, predict2, target)

        models_dataframe=pd.DataFrame(values,index=['Linear SVM', 'Radial SVM', 'Logistic Regression', 'KNN', 'Decision Tree', 'XGBOOST', 'LightGBM', 'model1', 'model2'])
        models_dataframe.columns=['Precision', 'Recall', 'F1-score', 'AUROC', 'Accuracy']
        print(models_dataframe)
