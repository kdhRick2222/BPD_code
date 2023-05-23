import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def normalize(input):
    return (input - input.min()) / (input.max() - input.min())


class TrainDataset_multi(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_multi, self).__init__()
        self.knnfile = knnfile
        self.df = pd.read_excel(self.knnfile)

        self.bpd1_num = len(self.df[self.df['bdp'] == 1])
        self.bpd2_num = len(self.df[self.df['bdp'] == 2])
        self.bpd3_num = len(self.df[self.df['bdp'] == 3])
        self.bpd4_num = len(self.df[self.df['bdp'] == 4])

        self.df_train = self.df.loc[:, :'rds']
        self.df_test = self.df.loc[:, 'bdp']
        self.df_train_norm = normalize(self.df_train)
        self.df_train_norm.fillna(0, inplace = True)
        self.df = self.df_train_norm.join(self.df_test)

        self.bpd1 = self.df['bdp'] == 1
        self.df1 = self.df[self.bpd1]
        self.np_df1 = self.df1.values

        self.bpd2 = self.df['bdp'] == 2
        self.df2 = self.df[self.bpd2]
        self.np_df2 = self.df2.values

        self.bpd3 = self.df['bdp'] == 3
        self.df3 = self.df[self.bpd3]
        self.np_df3 = self.df3.values

        self.bpd4 = self.df['bdp'] == 4
        self.df4 = self.df[self.bpd4]
        self.np_df4 = self.df4.values

        # print("NUMBERS : ", self.bpd1_num, self.bpd2_num, self.bpd3_num, self.bpd4_num)

        self.ratio = 0.9

        self.x_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), :45], self.np_df2[:int(self.bpd2_num*self.ratio), :45], self.np_df3[:int(self.bpd3_num*self.ratio), :45], 
                                        self.np_df4[:int(self.bpd4_num*self.ratio), :45]), axis=0)
        self.y_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), 45:], self.np_df2[:int(self.bpd2_num*self.ratio), 45:], self.np_df3[:int(self.bpd3_num*self.ratio), 45:], 
                                        self.np_df4[:int(self.bpd4_num*self.ratio), 45:]), axis=0)

    def __getitem__(self, idx):
        traindata = torch.FloatTensor(self.x_train1[idx])
        labeldata = torch.FloatTensor(self.y_train1[idx]) - 1
        # labeldata = F.one_hot(torch.FloatTensor(self.y_train1[idx]) - 1, num_class=4)
        return traindata, labeldata

    def __len__(self):
        return len(self.x_train1)


class EvalDataset_multi(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset_multi, self).__init__()
        self.knnfile = knnfile
        self.df = pd.read_excel(self.knnfile) #, usecols = self.data)

        self.bpd1_num = len(self.df[self.df['bdp'] == 1])
        self.bpd2_num = len(self.df[self.df['bdp'] == 2])
        self.bpd3_num = len(self.df[self.df['bdp'] == 3])
        self.bpd4_num = len(self.df[self.df['bdp'] == 4])

        self.df_train = self.df.loc[:, :'rds']
        self.df_test = self.df.loc[:, 'bdp']
        self.df_train_norm = normalize(self.df_train)
        self.df_train_norm.fillna(0, inplace = True)
        self.df = self.df_train_norm.join(self.df_test)

        self.bpd1 = self.df['bdp'] == 1
        self.df1 = self.df[self.bpd1]
        self.np_df1 = self.df1.values

        self.bpd2 = self.df['bdp'] == 2
        self.df2 = self.df[self.bpd2]
        self.np_df2 = self.df2.values

        self.bpd3 = self.df['bdp'] == 3
        self.df3 = self.df[self.bpd3]
        self.np_df3 = self.df3.values

        self.bpd4 = self.df['bdp'] == 4
        self.df4 = self.df[self.bpd4]
        self.np_df4 = self.df4.values

        self.ratio = 0.9

        self.x_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, :45], self.np_df2[int(self.bpd2_num*self.ratio):, :45], self.np_df3[int(self.bpd3_num*self.ratio):, :45], 
                                       self.np_df4[int(self.bpd4_num*self.ratio):, :45]), axis=0)
        self.y_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, 45:], self.np_df2[int(self.bpd2_num*self.ratio):, 45:], self.np_df3[int(self.bpd3_num*self.ratio):, 45:], 
                                       self.np_df4[int(self.bpd4_num*self.ratio):, 45:]), axis=0)

    def __getitem__(self, idx):
        testdata = torch.FloatTensor(self.x_test1[idx])
        labeldata = torch.FloatTensor(self.y_test1[idx]) - 1
        # labeldata = F.one_hot(torch.FloatTensor(self.y_test1[idx]) - 1, num_class=4)
        return testdata, labeldata

    def __len__(self):
        return len(self.x_test1)



class TrainDataset_bi(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_bi, self).__init__()
        self.knnfile = knnfile

        self.df = pd.read_excel(self.knnfile) #, index_col = 0) #, usecols = self.data)
        # self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x== 1 else 2)
        # self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x < 3 else 2)
        self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x < 4 else 2)

        self.bpd1_num = len(self.df[self.df['bdp'] == 1])
        self.bpd2_num = len(self.df[self.df['bdp'] == 2])

        self.df_train = self.df.loc[:, :'rds']
        self.df_test = self.df.loc[:, 'bdp']
        self.df_train_norm = normalize(self.df_train)
        self.df_train_norm.fillna(0, inplace = True)
        self.df = self.df_train_norm.join(self.df_test)

        self.bpd1 = self.df['bdp'] == 1
        self.df1 = self.df[self.bpd1]
        self.np_df1 = self.df1.values

        self.bpd2 = self.df['bdp'] == 2
        self.df2 = self.df[self.bpd2]
        self.np_df2 = self.df2.values

        self.ratio = 0.9

        self.x_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), :45], self.np_df2[:int(self.bpd2_num*self.ratio), :45]), axis=0)
        self.y_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), 45:], self.np_df2[:int(self.bpd2_num*self.ratio), 45:]), axis=0)

    def __getitem__(self, idx):
        traindata = torch.FloatTensor(self.x_train1[idx])
        labeldata = torch.FloatTensor(self.y_train1[idx]) - 1
        return traindata, labeldata

    def __len__(self):
        return len(self.x_train1)


class EvalDataset_bi(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset_bi, self).__init__()
        self.knnfile = knnfile
        self.df = pd.read_excel(self.knnfile) #, index_col = 0) #, usecols = self.data)
        # self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x== 1 else 2)
        # self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x < 3 else 2)
        self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x < 4 else 2)

        self.bpd1_num = len(self.df[self.df['bdp'] == 1])
        self.bpd2_num = len(self.df[self.df['bdp'] == 2])

        self.df_train = self.df.loc[:, :'rds']
        self.df_test = self.df.loc[:, 'bdp']
        self.df_train_norm = normalize(self.df_train)
        self.df_train_norm.fillna(0, inplace = True)
        self.df = self.df_train_norm.join(self.df_test)

        self.bpd1 = self.df['bdp'] == 1
        self.df1 = self.df[self.bpd1]
        self.np_df1 = self.df1.values

        self.bpd2 = self.df['bdp'] == 2
        self.df2 = self.df[self.bpd2]
        self.np_df2 = self.df2.values

        self.ratio = 0.9

        self.x_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, :45], self.np_df2[int(self.bpd2_num*self.ratio):, :45]), axis=0)
        self.y_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, 45:], self.np_df2[int(self.bpd2_num*self.ratio):, 45:]), axis=0)

    def __getitem__(self, idx):
        testdata = torch.FloatTensor(self.x_test1[idx])
        labeldata = torch.FloatTensor(self.y_test1[idx]) - 1
        return testdata, labeldata

    def __len__(self):
        return len(self.x_test1)


############################################


class TrainDataset_multi_rds(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_multi_rds, self).__init__()
        self.knnfile = knnfile

        self.df = pd.read_excel(self.knnfile) #, index_col = 0) #, usecols = self.data)
        self.rds2 = self.df['rds'] == 2 # 1 or 2
        self.df = self.df[self.rds2]

        self.bpd1_num = len(self.df[self.df['bdp'] == 1])
        self.bpd2_num = len(self.df[self.df['bdp'] == 2])
        self.bpd3_num = len(self.df[self.df['bdp'] == 3])
        self.bpd4_num = len(self.df[self.df['bdp'] == 4])

        self.df_train = self.df.loc[:, :'rds']
        self.df_test = self.df.loc[:, 'bdp']
        self.df_train_norm = normalize(self.df_train)
        self.df_train_norm.fillna(0, inplace = True)
        self.df = self.df_train_norm.join(self.df_test)

        self.bpd1 = self.df['bdp'] == 1
        self.df1 = self.df[self.bpd1]
        self.np_df1 = self.df1.values

        self.bpd2 = self.df['bdp'] == 2
        self.df2 = self.df[self.bpd2]
        self.np_df2 = self.df2.values

        self.bpd3 = self.df['bdp'] == 3
        self.df3 = self.df[self.bpd3]
        self.np_df3 = self.df3.values

        self.bpd4 = self.df['bdp'] == 4
        self.df4 = self.df[self.bpd4]
        self.np_df4 = self.df4.values

        self.ratio = 0.9

        self.x_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), :45], self.np_df2[:int(self.bpd2_num*self.ratio), :45], self.np_df3[:int(self.bpd3_num*self.ratio), :45], 
                                        self.np_df4[:int(self.bpd4_num*self.ratio), :45]), axis=0)
        self.y_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), 45:], self.np_df2[:int(self.bpd2_num*self.ratio), 45:], self.np_df3[:int(self.bpd3_num*self.ratio), 45:], 
                                        self.np_df4[:int(self.bpd4_num*self.ratio), 45:]), axis=0)

    def __getitem__(self, idx):
        traindata = torch.FloatTensor(self.x_train1[idx])
        labeldata = torch.FloatTensor(self.y_train1[idx]) - 1
        # labeldata = F.one_hot(torch.FloatTensor(self.y_train1[idx]) - 1, num_class=4)
        return traindata, labeldata

    def __len__(self):
        return len(self.x_train1)


class EvalDataset_multi_rds(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset_multi_rds, self).__init__()
        self.knnfile = knnfile
        self.df = pd.read_excel(self.knnfile)
        self.rds2 = self.df['rds'] == 2
        self.df = self.df[self.rds2]

        self.bpd1_num = len(self.df[self.df['bdp'] == 1])
        self.bpd2_num = len(self.df[self.df['bdp'] == 2])
        self.bpd3_num = len(self.df[self.df['bdp'] == 3])
        self.bpd4_num = len(self.df[self.df['bdp'] == 4])

        self.df_train = self.df.loc[:, :'rds']
        self.df_test = self.df.loc[:, 'bdp']
        self.df_train_norm = normalize(self.df_train)
        self.df_train_norm.fillna(0, inplace = True)
        self.df = self.df_train_norm.join(self.df_test)

        self.bpd1 = self.df['bdp'] == 1
        self.df1 = self.df[self.bpd1]
        self.np_df1 = self.df1.values

        self.bpd2 = self.df['bdp'] == 2
        self.df2 = self.df[self.bpd2]
        self.np_df2 = self.df2.values

        self.bpd3 = self.df['bdp'] == 3
        self.df3 = self.df[self.bpd3]
        self.np_df3 = self.df3.values

        self.bpd4 = self.df['bdp'] == 4
        self.df4 = self.df[self.bpd4]
        self.np_df4 = self.df4.values

        self.ratio = 0.9

        self.x_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, :45], self.np_df2[int(self.bpd2_num*self.ratio):, :45], self.np_df3[int(self.bpd3_num*self.ratio):, :45], 
                                       self.np_df4[int(self.bpd4_num*self.ratio):, :45]), axis=0)
        self.y_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, 45:], self.np_df2[int(self.bpd2_num*self.ratio):, 45:], self.np_df3[int(self.bpd3_num*self.ratio):, 45:], 
                                       self.np_df4[int(self.bpd4_num*self.ratio):, 45:]), axis=0)

    def __getitem__(self, idx):
        testdata = torch.FloatTensor(self.x_test1[idx])
        labeldata = torch.FloatTensor(self.y_test1[idx]) - 1
        # labeldata = F.one_hot(torch.FloatTensor(self.y_test1[idx]) - 1, num_class=4)
        return testdata, labeldata

    def __len__(self):
        return len(self.x_test1)



class TrainDataset_bi_rds(Dataset):
    def __init__(self, knnfile):
        super(TrainDataset_bi_rds, self).__init__()
        self.knnfile = knnfile

        self.df = pd.read_excel(self.knnfile) #, index_col = 0) #, usecols = self.data)
        # self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x== 1 else 2)
        # self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x < 3 else 2)
        self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x < 4 else 2)
        self.rds2 = self.df['rds'] == 2
        self.df = self.df[self.rds2]

        self.bpd1_num = len(self.df[self.df['bdp'] == 1])
        self.bpd2_num = len(self.df[self.df['bdp'] == 2])

        self.df_train = self.df.loc[:, :'rds']
        self.df_test = self.df.loc[:, 'bdp']
        self.df_train_norm = normalize(self.df_train)
        self.df_train_norm.fillna(0, inplace = True)
        self.df = self.df_train_norm.join(self.df_test)

        self.bpd1 = self.df['bdp'] == 1
        self.df1 = self.df[self.bpd1]
        self.np_df1 = self.df1.values

        self.bpd2 = self.df['bdp'] == 2
        self.df2 = self.df[self.bpd2]
        self.np_df2 = self.df2.values

        self.ratio = 0.9

        self.x_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), :45], self.np_df2[:int(self.bpd2_num*self.ratio), :45]), axis=0)
        self.y_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), 45:], self.np_df2[:int(self.bpd2_num*self.ratio), 45:]), axis=0)

    def __getitem__(self, idx):
        traindata = torch.FloatTensor(self.x_train1[idx])
        labeldata = torch.FloatTensor(self.y_train1[idx]) - 1
        return traindata, labeldata

    def __len__(self):
        return len(self.x_train1)


class EvalDataset_bi_rds(Dataset):
    def __init__(self, knnfile):
        super(EvalDataset_bi_rds, self).__init__()
        self.knnfile = knnfile

        self.df = pd.read_excel(self.knnfile) #, index_col = 0) #, usecols = self.data)
        # self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x== 1 else 2)
        # self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x < 3 else 2)
        self.df['bdp'] = self.df['bdp'].apply(lambda x: 1 if x < 4 else 2)
        self.rds2 = self.df['rds'] == 2
        self.df = self.df[self.rds2]

        self.bpd1_num = len(self.df[self.df['bdp'] == 1])
        self.bpd2_num = len(self.df[self.df['bdp'] == 2])

        self.df_train = self.df.loc[:, :'rds']
        self.df_test = self.df.loc[:, 'bdp']
        self.df_train_norm = normalize(self.df_train)
        self.df_train_norm.fillna(0, inplace = True)
        self.df = self.df_train_norm.join(self.df_test)

        self.bpd1 = self.df['bdp'] == 1
        self.df1 = self.df[self.bpd1]
        self.np_df1 = self.df1.values

        self.bpd2 = self.df['bdp'] == 2
        self.df2 = self.df[self.bpd2]
        self.np_df2 = self.df2.values

        self.ratio = 0.9

        self.x_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, :45], self.np_df2[int(self.bpd2_num*self.ratio):, :45]), axis=0)
        self.y_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, 45:], self.np_df2[int(self.bpd2_num*self.ratio):, 45:]), axis=0)

    def __getitem__(self, idx):
        testdata = torch.FloatTensor(self.x_test1[idx])
        labeldata = torch.FloatTensor(self.y_test1[idx]) - 1
        return testdata, labeldata

    def __len__(self):
        return len(self.x_test1)



# class TrainDataset_multi_arre28(Dataset):
#     def __init__(self, knnfile):
#         super(TrainDataset_multi_arre28, self).__init__()
#         self.knnfile = knnfile
#         self.df = pd.read_excel(self.knnfile)

#         self.bpd1_num = len(self.df[self.df['arre28'] == 0])
#         self.bpd2_num = len(self.df[self.df['arre28'] == 1])
#         self.bpd3_num = len(self.df[self.df['arre28'] == 2])

#         self.df_train = self.df.loc[:, :'rds']
#         self.df_test = self.df.loc[:, 'arre28']
#         self.df_train_norm = normalize(self.df_train)
#         self.df_train_norm.fillna(0, inplace = True)
#         self.df = self.df_train_norm.join(self.df_test)

#         self.bpd1 = self.df['arre28'] == 0
#         self.df1 = self.df[self.bpd1]
#         self.np_df1 = self.df1.values

#         self.bpd2 = self.df['arre28'] == 1
#         self.df2 = self.df[self.bpd2]
#         self.np_df2 = self.df2.values

#         self.bpd3 = self.df['arre28'] == 2
#         self.df3 = self.df[self.bpd3]
#         self.np_df3 = self.df3.values

#         self.ratio = 0.9

#         self.x_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), :60], self.np_df2[:int(self.bpd2_num*self.ratio), :60], self.np_df3[:int(self.bpd3_num*self.ratio), :60]), axis=0)
#         self.y_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), 60:], self.np_df2[:int(self.bpd2_num*self.ratio), 60:], self.np_df3[:int(self.bpd3_num*self.ratio), 60:]), axis=0)

#     def __getitem__(self, idx):
#         traindata = torch.FloatTensor(self.x_train1[idx])
#         labeldata = torch.FloatTensor(self.y_train1[idx])
#         # labeldata = F.one_hot(torch.FloatTensor(self.y_train1[idx]) - 1, num_class=4)
#         return traindata, labeldata

#     def __len__(self):
#         return len(self.x_train1)


# class EvalDataset_multi_arre28(Dataset):
#     def __init__(self, knnfile):
#         super(EvalDataset_multi_arre28, self).__init__()
#         self.knnfile = knnfile
#         self.df = pd.read_excel(self.knnfile) #, usecols = self.data)

#         self.bpd1_num = len(self.df[self.df['arre28'] == 0])
#         self.bpd2_num = len(self.df[self.df['arre28'] == 1])
#         self.bpd3_num = len(self.df[self.df['arre28'] == 2])

#         self.df_train = self.df.loc[:, :'rds']
#         self.df_test = self.df.loc[:, 'arre28']
#         self.df_train_norm = normalize(self.df_train)
#         self.df_train_norm.fillna(0, inplace = True)
#         self.df = self.df_train_norm.join(self.df_test)

#         self.bpd1 = self.df['arre28'] == 0
#         self.df1 = self.df[self.bpd1]
#         self.np_df1 = self.df1.values

#         self.bpd2 = self.df['arre28'] == 1
#         self.df2 = self.df[self.bpd2]
#         self.np_df2 = self.df2.values

#         self.bpd3 = self.df['arre28'] == 2
#         self.df3 = self.df[self.bpd3]
#         self.np_df3 = self.df3.values

#         self.ratio = 0.9

#         self.x_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, :60], self.np_df2[int(self.bpd2_num*self.ratio):, :60], self.np_df3[int(self.bpd3_num*self.ratio):, :60]), axis=0)
#         self.y_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, 60:], self.np_df2[int(self.bpd2_num*self.ratio):, 60:], self.np_df3[int(self.bpd3_num*self.ratio):, 60:]), axis=0)

#     def __getitem__(self, idx):
#         testdata = torch.FloatTensor(self.x_test1[idx])
#         labeldata = torch.FloatTensor(self.y_test1[idx])
#         # labeldata = F.one_hot(torch.FloatTensor(self.y_test1[idx]) - 1, num_class=4)
#         return testdata, labeldata

#     def __len__(self):
#         return len(self.x_test1)



# class TrainDataset_multi_arre36(Dataset):
#     def __init__(self, knnfile):
#         super(TrainDataset_multi_arre36, self).__init__()
#         self.knnfile = knnfile
#         self.df = pd.read_excel(self.knnfile)

#         self.bpd1_num = len(self.df[self.df['arre36'] == 0])
#         self.bpd2_num = len(self.df[self.df['arre36'] == 1])
#         self.bpd3_num = len(self.df[self.df['arre36'] == 2])

#         self.df_train = self.df.loc[:, :'rds']
#         self.df_test = self.df.loc[:, 'arre36']
#         self.df_train_norm = normalize(self.df_train)
#         self.df_train_norm.fillna(0, inplace = True)
#         self.df = self.df_train_norm.join(self.df_test)

#         self.bpd1 = self.df['arre36'] == 0
#         self.df1 = self.df[self.bpd1]
#         self.np_df1 = self.df1.values

#         self.bpd2 = self.df['arre36'] == 1
#         self.df2 = self.df[self.bpd2]
#         self.np_df2 = self.df2.values

#         self.bpd3 = self.df['arre36'] == 2
#         self.df3 = self.df[self.bpd3]
#         self.np_df3 = self.df3.values

#         self.ratio = 0.9

#         self.x_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), :45], self.np_df2[:int(self.bpd2_num*self.ratio), :45], self.np_df3[:int(self.bpd3_num*self.ratio), :45]), axis=0)
#         self.y_train1 = np.concatenate((self.np_df1[:int(self.bpd1_num*self.ratio), 45:], self.np_df2[:int(self.bpd2_num*self.ratio), 45:], self.np_df3[:int(self.bpd3_num*self.ratio), 45:]), axis=0)

#     def __getitem__(self, idx):
#         traindata = torch.FloatTensor(self.x_train1[idx])
#         labeldata = torch.FloatTensor(self.y_train1[idx])
#         # labeldata = F.one_hot(torch.FloatTensor(self.y_train1[idx]) - 1, num_class=4)
#         return traindata, labeldata

#     def __len__(self):
#         return len(self.x_train1)


# class EvalDataset_multi_arre36(Dataset):
#     def __init__(self, knnfile):
#         super(EvalDataset_multi_arre36, self).__init__()
#         self.knnfile = knnfile
#         self.df = pd.read_excel(self.knnfile) #, usecols = self.data)

#         self.bpd1_num = len(self.df[self.df['arre36'] == 0])
#         self.bpd2_num = len(self.df[self.df['arre36'] == 1])
#         self.bpd3_num = len(self.df[self.df['arre36'] == 2])

#         self.df_train = self.df.loc[:, :'rds']
#         self.df_test = self.df.loc[:, 'arre36']
#         self.df_train_norm = normalize(self.df_train)
#         self.df_train_norm.fillna(0, inplace = True)
#         self.df = self.df_train_norm.join(self.df_test)

#         self.bpd1 = self.df['arre36'] == 0
#         self.df1 = self.df[self.bpd1]
#         self.np_df1 = self.df1.values

#         self.bpd2 = self.df['arre36'] == 1
#         self.df2 = self.df[self.bpd2]
#         self.np_df2 = self.df2.values

#         self.bpd3 = self.df['arre36'] == 2
#         self.df3 = self.df[self.bpd3]
#         self.np_df3 = self.df3.values

#         self.ratio = 0.9

#         self.x_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, :45], self.np_df2[int(self.bpd2_num*self.ratio):, :45], self.np_df3[int(self.bpd3_num*self.ratio):, :45]), axis=0)
#         self.y_test1 = np.concatenate((self.np_df1[int(self.bpd1_num*self.ratio):, 45:], self.np_df2[int(self.bpd2_num*self.ratio):, 45:], self.np_df3[int(self.bpd3_num*self.ratio):, 45:]), axis=0)

#     def __getitem__(self, idx):
#         testdata = torch.FloatTensor(self.x_test1[idx])
#         labeldata = torch.FloatTensor(self.y_test1[idx])
#         # labeldata = F.one_hot(torch.FloatTensor(self.y_test1[idx]) - 1, num_class=4)
#         return testdata, labeldata

#     def __len__(self):
#         return len(self.x_test1)




