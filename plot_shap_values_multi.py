# import shap

# features = ['sftun', 'gagew', 'apgs1', 'apgs5', 'bwei', 'bbph', 'bbbe', 'mage',
#        'bhei', 'bhead', 'gran', 'parn', 'als', 'oligo', 'poly', 'ivh', 'white',
#        'mph', 'ph', 'sft', 'strdu', 'mulg', 'dm_o', 'dm_g', 'htn_o', 'htn_g',
#        'chor', 'prom', 'delm', 'sex', 'resu', 'eythtran', 'sga', 'prep',
#        'spda', 'pdat', 'pda_l', 'lbp', 'nese', 'ibif', 'meni', 'ntet', 'sip',
#        'rop', 'z_score', 'presteroid', 'rds']

# explainer = shap.Explainer(models[5])
# shap_values = explainer(x_train2)
# # shap_values = explainer(df_new)
# # shap.plots.beeswarm(shap_values, max_display=20)
# shap.summary_plot(shap_values, x_train2, max_display=10, feature_names=features)


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
import shap

from KNN_model import *
from data_loader_multi import *
from earlystop import EarlyStopping
from sklearn.preprocessing import label_binarize

##
# python plot_shap_values.py --model_weight '' --target ''

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, default="/home/daehyun/mnt/nas12/KNN_bpd/weight_final/model2_bpd_multi_7855.pth")
    parser.add_argument('--data', type=str, default="./knn_preprocessed.xlsx")
    parser.add_argument('--target', type=str, default="bpd_multi_model2")
    parser.add_argument('--refer', type=str, default="supportv1")
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    target = args.target

    
    if target == 'bpd_multi_model2':
        model = KNN_bpd_multi_model2().to(device)
        train_dataset = TrainDataset_multi_model2(args.data, args.refer)
        eval_dataset = EvalDataset_multi_model2(args.data, args.refer)
        model.load_state_dict(torch.load(args.model_weight))

    elif target == 'bpd_bi_model2':
        model = KNN_bpd_bi_model2().to(device)
        train_dataset = TrainDataset_bi_model2(args.data, args.refer)
        eval_dataset = EvalDataset_bi_model2(args.data, args.refer)
        model.load_state_dict(torch.load(args.model_weight))

    elif target == 'rds_multi_model2':
        model = KNN_bpd_multi_model2().to(device)
        train_dataset = TrainDataset_multi_rds_model2(args.data, args.refer)
        eval_dataset = EvalDataset_multi_rds_model2(args.data, args.refer)
        model.load_state_dict(torch.load(args.model_weight))

    elif target == 'rds_bi_model2':
        model = KNN_bpd_bi_model2().to(device)
        train_dataset = TrainDataset_bi_rds_model2(args.data, args.refer)
        eval_dataset = EvalDataset_bi_rds_model2(args.data, args.refer)
        model.load_state_dict(torch.load(args.model_weight))

    else:
        print('There is no target named like ', target)


    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=1000,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=False,
                                  drop_last=True)

    for data in train_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        print('one')
        break
    
    # if 'bpd' in target:
    #     features = ['sftun', 'gage', 'apgs1', 'apgs5', 'bwei', 'bbph', 'birthbe', 'mage',
    #                 'bhei', 'bhead', 'gran', 'parn', 'als', 'oligo', 'poly', 'ivh', 'mph',
    #                 'sft', 'mulg', 'dm_o', 'dm_g', 'htn_o', 'htn_g', 'chor', 'prom', 'delm',
    #                 'sex', 'resu', 'earlysep', 'earlyfung', 'sga', 'prep', 'spda', 'pdat',
    #                 'pda_l_7', 'pda_l', 'lbp', 'ibif', 'resuegrade', 'sftstartday', 'ph7d',
    #                 'earlymeni', 'z_score', 'presteroid']
    # else:
    #     features = ['sftun', 'gage', 'apgs1', 'apgs5', 'bwei', 'bbph', 'birthbe', 'mage',
    #                 'bhei', 'bhead', 'gran', 'parn', 'als', 'oligo', 'poly', 'ivh', 'mph',
    #                 'sft', 'mulg', 'dm_o', 'dm_g', 'htn_o', 'htn_g', 'chor', 'prom', 'delm',
    #                 'sex', 'resu', 'earlysep', 'earlyfung', 'sga', 'prep', 'spda', 'pdat',
    #                 'pda_l_7', 'pda_l', 'lbp', 'ibif', 'resuegrade', 'sftstartday', 'ph7d',
    #                 'earlymeni', 'z_score', 'presteroid', 'rds']

    # features = ['sftun', 'gage', 'apgs1', 'apgs5', 'bwei', 'bbph', 'birthbe', 'btem',
    #             'mage', 'bhei', 'bhead', 'gran', 'parn', 'als', 'oligo', 'poly', 'ivh',
    #             'mph', 'sft', 'mulg', 'dm_o', 'dm_g', 'htn_o', 'htn_g', 'chor', 'prom',
    #             'delm', 'sex', 'resu', 'earlysep', 'earlyfung', 'sga', 'prep', 'spda',
    #             'pdat', 'pda_l_7', 'pda_l', 'lbp', 'ibif', 'resuegrade', 'ph7d', 
    #             'earlymeni', 'z_score', 'presteroid', 'rds']

    features = ['SFTnu', 'GA', 'AS1', 'AS5', 'BW', 'pH1h', 'BE1h', 'BTEM', 'M_AGE', 
                'BHt', 'BHC', 'GRAV', 'PARI', 'ALS', 'OLIG', 'POLY', 'IVH', 'PHem',
                'SFT', 'MULT', 'O_DM', 'G_DM', 'HTN', 'PIH', 'CA', 'PROM', 'C-SEC', 
                'sex', 'RESU', 'SEPS', 'FUNG', 'SGA', 'IVF', 'PDAs', 'PDATx', 'PDALg7', 
                'PDALg', 'lowBP', 'C_INF', 'RESGr', 'PHT', 'MENI', 'BWt_z', 'PRE_S', 'RDS']


    np_inputs = t2np(inputs)

    # print(model)
    explainer_shap = shap.DeepExplainer(model, inputs) 
    shap_values = explainer_shap.shap_values(inputs)
    shap.summary_plot(shap_values, np_inputs, feature_names=features)

    # explainer_shap1 = shap.DeepExplainer((model.bpd1, model.refer1, model.bpd2), inputs)
    # shap_values1 = explainer_shap1.shap_values(inputs)
    # shap.summary_plot(shap_values1, np_inputs, feature_names=features)

    import matplotlib.pyplot as pl
    pl.savefig("./shap_pic/"+target+"_summary_plot.pdf")