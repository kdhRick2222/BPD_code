import numpy as np
import pandas as pd
from data_loader_classicalmethods import *

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, f1_score
from sklearn.preprocessing import Binarizer
from numpy import argmax
from sklearn.preprocessing import label_binarize


path = '/home/daehyun/mnt/nas12/KNN_bpd/knn_preprocessed.xlsx'
query = 'rds2_multi'



def get_clf_eval(y_test, pred):
       # confusion = confusion_matrix(y_test, pred)
       ppv = precision_score(y_test, pred)#, average='macro')
       # npv = precision_score(1-y_test, 1-pred)
       sensitivity = recall_score(y_test, pred)#, average='macro')
       # f1_score_1 = f1_score(label1, custom_predict1, labels=None, average='macro')
       # f1 = f1_score(y_test, pred, labels=None)
       return ppv, sensitivity

def get_clf_eval_for_multi(y_test, pred):
       ppv = precision_score(y_test, pred, average='macro')
       # npv = precision_score(1-y_test, 1-pred)
       sensitivity = recall_score(y_test, pred, average='macro')
       # f1_score_1 = f1_score(label1, custom_predict1, labels=None, average='macro')
       # f1 = f1_score(y_test, pred, labels=None)
       return ppv, sensitivity


if query == 'bpd_multi':
       train = TrainDataset_bpd_multi_np(path)
       test = EvalDataset_bpd_multi_np(path)
elif 'bpd_bi' in query:
       train = TrainDataset_bpd_bi_np(path, query)
       test = EvalDataset_bpd_bi_np(path, query)
elif (query == 'rds1_multi') or (query == 'rds2_multi'):
       train = TrainDataset_multi_rds_np(path, query)
       test = EvalDataset_multi_rds_np(path, query)
elif ('rds' in query) and ('bi' in query):
       train = TrainDataset_bi_rds_np(path, query)
       test = EvalDataset_bi_rds_np(path, query)
else:
       print("There is no Dataloader of the target...")


if 'multi' in query:
       x_test = test[:][0]
       y_test = test[:][1]
       x_train = train[:][0]
       y_train = train[:][1]
       # print(y_train.shape)
       y_train_ = label_binarize(y_train, classes=[0, 1, 2, 3])
       y_test_ = label_binarize(y_test, classes=[0, 1, 2, 3])
else:
       x_test = test[:][0]
       y_test = test[:][1].reshape(-1)
       x_train = train[:][0]
       y_train = train[:][1].reshape(-1)


abc=[]
classifiers=['Linear SVM', 'Radial SVM', 'Logistic Regression', 'KNN', 'Decision Tree', 'XGBOOST', 'LightGBM', 'Random Forest']
models=[svm.SVC(kernel='linear', probability=True),svm.SVC(kernel='rbf', probability=True),LogisticRegression(),KNeighborsClassifier(),
       DecisionTreeClassifier(),XGBClassifier(use_label_encoder=False, eval_metric='logloss'),LGBMClassifier(),RandomForestClassifier()]

if 'multi' in query:
       for i in models:
              model = i
              # print(x_train.shape, y_train.shape) # (12431, 46) (12431, 1)
              model.fit(x_train, y_train.ravel())
              prediction=model.predict_proba(x_test)
              print(y_test_.shape, prediction.shape) # ()
              auroc = roc_auc_score(y_test_, prediction, multi_class='ovo')
              accuracy = accuracy_score(y_test, np.argmax(prediction, 1))

              # fpr, tpr, thresholds = roc_curve(y_test, prediction)
              # J = tpr - fpr
              # ix = argmax(J)
              # best_threshold = thresholds[ix]
              predict1 = np.array(prediction)
              label = np.array(y_test)
              # binarizer = Binarizer(threshold=best_threshold)
              # custom_predict = binarizer.fit_transform(predict1.reshape(-1, 1))
              # ppv, sensitivity = get_clf_eval(label, custom_predict)
              ppv, sensitivity = get_clf_eval_for_multi(label, np.argmax(prediction, 1))
              f1 = f1_score(label, np.argmax(prediction, 1), average='macro')

              abc.append([ppv, sensitivity, f1, auroc, accuracy])
else:
       for i in models:
              model = i
              model.fit(x_train, y_train.ravel())
              prediction = model.predict_proba(x_test)

              prediction = np.argmax(np.array(prediction), 1)

              fpr, tpr, thresholds = roc_curve(y_test, prediction)
              J = tpr - fpr
              ix = argmax(J)
              best_threshold = thresholds[ix]
              predict1 = np.array(prediction)
              label = np.array(y_test)
              binarizer = Binarizer(threshold=best_threshold)
              custom_predict = binarizer.fit_transform(predict1.reshape(-1, 1))
              ppv, sensitivity = get_clf_eval(label, predict1)

              f1 = f1_score(label, predict1)#, average='macro')
              auroc = roc_auc_score(y_test, prediction)
              # print(y_test.shape, prediction.shape)
              accuracy = accuracy_score(y_test, predict1)

              abc.append([ppv, sensitivity, f1, auroc, accuracy])


models_dataframe=pd.DataFrame(abc,index=classifiers)
models_dataframe.columns=['precision', 'recall', 'F1-score', 'AUROC', 'accuracy']
print(query)
print(models_dataframe)



