import torch
from torch import nn
import math


class KNN_bpd_multi(nn.Module):
    def __init__(self):
        super(KNN_bpd_multi, self).__init__()

        # self.sq = nn.Sequential(
        #     nn.Linear(46, 32), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(32), 
        #     # nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(32), 
        #     nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(16), 
        #     nn.Linear(16, 8), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(8), 
        #     nn.Linear(8, 4), nn.Softmax(dim=1),)
        self.sq = nn.Sequential(
            nn.Linear(45, 28), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(28),
            nn.Linear(28, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12),
            nn.Linear(12, 4), nn.Softmax(dim=1),)

    def forward(self, x):
        x = self.sq(x)
        return x


class KNN_rds_multi(nn.Module):
    def __init__(self):
        super(KNN_rds_multi, self).__init__()

        # self.sq = nn.Sequential(
        #     nn.Linear(46, 32), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(32), 
        #     nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(16), 
        #     nn.Linear(16, 8), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(8), 
        #     nn.Linear(8, 4), nn.Softmax(dim=1),)
        self.sq = nn.Sequential(
            nn.Linear(45, 28), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(28),
            nn.Linear(28, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12),
            nn.Linear(12, 4), nn.Softmax(dim=1),)

    def forward(self, x):
        x = self.sq(x)
        return x


class KNN_bpd_bi(nn.Module):
    def __init__(self):
        super(KNN_bpd_bi, self).__init__()

        # self.sq = nn.Sequential(
        #     nn.Linear(46, 16), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(16), 
        #     nn.Linear(16, 8), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(8), 
        #     nn.Linear(8, 4), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(4), 
        #     nn.Linear(4, 1), nn.Sigmoid(),)
        self.sq = nn.Sequential(
            nn.Linear(45, 28), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(28),
            nn.Linear(28, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12),
            nn.Linear(12, 1), nn.Sigmoid(),)

    def forward(self, x):
        x = self.sq(x)
        return x


class KNN_rds_bi(nn.Module):
    def __init__(self):
        super(KNN_rds_bi, self).__init__()
        
        # self.sq = nn.Sequential(
        #     nn.Linear(46, 16), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(16), 
        #     nn.Linear(16, 8), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(8), 
        #     nn.Linear(8, 4), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(4), 
        #     nn.Linear(4, 1), nn.Sigmoid(),)
        self.sq = nn.Sequential(
            nn.Linear(45, 28), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(28),
            nn.Linear(28, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12),
            nn.Linear(12, 1), nn.Sigmoid(),)

    def forward(self, x):
        x = self.sq(x)
        return x


class KNN_arre_multi(nn.Module):
    def __init__(self):
        super(KNN_arre_multi, self).__init__()

        self.sq = nn.Sequential(
            nn.Linear(45, 32), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(32), 
            # nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(32), 
            nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(16), 
            nn.Linear(16, 8), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(8), 
            nn.Linear(8, 3), nn.Softmax(dim=1),)

    def forward(self, x):
        x = self.sq(x)
        return x

############################################################################################################

class KNN_bpd_multi_model2(nn.Module):
    def __init__(self):
        super(KNN_bpd_multi_model2, self).__init__()
        # ELU, 28, 12, 4

        self.refer1 = nn.Sequential(
            nn.Linear(45, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12), 
            # nn.Linear(28, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12),
            nn.Linear(12, 4), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(4),)
        
        self.refer2 = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid(),)

        self.bpd1 = nn.Sequential(
            nn.Linear(45, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12), 
            # nn.Linear(28, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12),
            nn.Linear(12, 4), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(4),)
        
        self.bpd2 = nn.Sequential(nn.Linear(8, 4), nn.Softmax(dim=1),)

    def forward(self, x):

        y = x
        x = self.refer1(x)
        y = self.bpd1(y)
        y = self.bpd2(torch.cat((x, y), dim=1))
        x = self.refer2(x)

        # return x, y
        return y



class KNN_bpd_bi_model2(nn.Module):
    def __init__(self):
        super(KNN_bpd_bi_model2, self).__init__()

        # self.refer1 = nn.Sequential(
        #     nn.Linear(46, 16), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(16), 
        #     nn.Linear(16, 8), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(8), 
        #     nn.Linear(8, 4), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(4),)
        
        # self.refer2 = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid(),)

        # self.bpd1 = nn.Sequential(
        #     nn.Linear(46, 16), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(16), 
        #     nn.Linear(16, 8), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(8), 
        #     nn.Linear(8, 4), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(4),)
        
        # self.bpd2 = nn.Sequential(
        #     nn.Linear(8, 4), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(4), 
        #     nn.Linear(4, 1), nn.Sigmoid(),)
        
        self.refer1 = nn.Sequential(
            nn.Linear(45, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12), 
            # nn.Linear(28, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12), 
            nn.Linear(12, 4), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(4),)
        
        self.refer2 = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid(),)

        self.bpd1 = nn.Sequential(
            nn.Linear(45, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12), 
            # nn.Linear(28, 12), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(12), 
            nn.Linear(12, 4), nn.ELU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(4),)
        
        self.bpd2 = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid(),)

    def forward(self, x):

        y = x

        x = self.refer1(x)
        y = self.bpd1(y)
        y = torch.cat((x, y), dim=1)
        y = self.bpd2(y)
        x = self.refer2(x)

        # return x, y
        return y



# class KNN_bpd_bi_model2(nn.Module):
#     def __init__(self):
#         super(KNN_bpd_multi_model2, self).__init__()

#         self.linear_a1 = nn.Linear(44, 32)
#         self.linear_a2 = nn.Linear(32, 16)
#         self.linear_a3 = nn.Linear(16, 4)
#         self.linear_a4 = nn.Linear(4, 1)

#         self.linear_b1 = nn.Linear(44, 32)
#         self.linear_b2 = nn.Linear(32, 16) 
#         self.linear_b3 = nn.Linear(16, 4)
#         self.linear_b4 = nn.Linear(8, 4)
#         self.linear_b5 = nn.Linear(4, 1)

#         self.bn64 = nn.BatchNorm1d(64)
#         self.bn32 = nn.BatchNorm1d(32)
#         self.bn16 = nn.BatchNorm1d(16)
#         self.bn8 = nn.BatchNorm1d(8)
#         self.bn4 = nn.BatchNorm1d(4)

#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(0.2)
#         self.sigmoid = nn.Sigmoid()

#         # self.sq1 = nn.Sequential(
#         #     nn.Linear(45, 32), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(32), 
#         #     nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(16), 
#         #     nn.Linear(16, 8), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.BatchNorm1d(8), 
#         #     nn.Linear(8, 3), nn.Softmax(dim=1),)

#     def forward(self, x):

#         y = x

#         x = self.bn32(self.dropout(self.relu(self.linear_a1(x))))
#         x = self.bn16(self.dropout(self.relu(self.linear_a2(x))))
#         x = self.bn4(self.dropout(self.relu(self.linear_a3(x))))

#         y = self.bn32(self.dropout(self.relu(self.linear_b1(y))))
#         y = self.bn16(self.dropout(self.relu(self.linear_b2(y))))
#         y = self.bn4(self.dropout(self.relu(self.linear_b3(y))))
#         y = torch.cat((x, y), dim=1)
#         y = self.bn4(self.dropout(self.relu(self.linear_b4(y))))
#         y = self.sigmoid(self.linear_b5(y))

#         x = self.sigmoid(self.linear_a4(x))

#         return x, y


