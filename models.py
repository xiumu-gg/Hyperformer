import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import torch.nn as nn
from math import sqrt
import time
import math


def circular_padding_chw(batch, padding):
    upper_pad = batch[..., -padding:, :]
    lower_pad = batch[..., :padding, :]
    temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

    left_pad = temp[..., -padding:]
    right_pad = temp[..., :padding]
    padded = torch.cat([left_pad, temp, right_pad], dim=3)
    return padded


class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)


class ConvR(BaseClass):
    def __init__(self, logger, dataset, max_arity, **kwargs):
        super(ConvR, self).__init__()
        self.device = kwargs['device']
        self.logger = logger
        self.has_run_debug = True
        self.max_arity = max_arity
        self.inp_drop_rate = kwargs['inp_drop']
        self.feature_map_drop_rate = kwargs['feature_map_drop']
        self.hidden_drop_rate = kwargs['hidden_drop']

        self.emb_dim = kwargs['emb_dim']
        self.reshape = kwargs['reshape']
        self.E = torch.nn.Embedding(dataset.num_ent(), self.emb_dim, padding_idx=0).to(self.device)

        self.rel_num = kwargs['R_internal_dim']
        self.kernel_size = kwargs['kernel_size']
        self.R_num = int(self.rel_num / (self.kernel_size[0] * self.kernel_size[1]))
        self.logger.debug(f'self.R_num={self.R_num}')
        self.R = torch.nn.Embedding(dataset.num_rel(), self.rel_num, padding_idx=0).to(self.device)

        self.bn0 = torch.nn.BatchNorm2d(1).to(self.device)
        self.bn1 = torch.nn.BatchNorm2d(self.R_num).to(self.device)
        self.bn2 = torch.nn.BatchNorm1d(self.rel_num).to(self.device)

        self.input_drop = torch.nn.Dropout(self.inp_drop_rate).to(self.device)
        self.feature_map_drop = torch.nn.Dropout(self.feature_map_drop_rate).to(self.device)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate).to(self.device)

        self.filtered = [self.reshape[0] - self.kernel_size[0] + 1, self.reshape[1] - self.kernel_size[1] + 1]
        self.p = torch.rand((self.filtered[0] * self.filtered[1], 1)).to(self.device)
        self.init()
        self.logger.debug(f'self.E.weight.data[0]={self.E.weight.data[0]}')

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim).to(self.device)
        self.R.weight.data[0] = torch.ones(self.rel_num).to(self.device)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])

    def convolve_in(self, r_idx, e_idx):
        batch_size = len(e_idx)
        e = self.E(e_idx).view(batch_size, 1, *self.reshape)
        self.logger.debug(f'The embedded dimension of the entity matrix is {e.size()}') if self.has_run_debug else 1
        e = self.bn0(e).view(1, batch_size, *self.reshape)
        e = self.input_drop(e)

        r = self.R(r_idx)
        r = self.bn2(r)
        r = self.input_drop(r)
        r = r.view(self.R_num * batch_size, 1, *self.kernel_size)

        self.logger.debug(f'The embedded dimension of the relation matrix is {r.size()}') if self.has_run_debug else 1
        self.has_run_debug = False

        x = F.conv2d(e, r, groups=batch_size)
        x = x.view(batch_size, self.R_num, *self.filtered)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.sum(dim=1)
        x = x.view(batch_size, -1)
        # x = torch.softmax(x, dim=1)
        for idd, i in enumerate(e_idx):
            if i == 0:
                x[idd] = torch.tensor(torch.tensor(np.ones((1, self.filtered[0] * self.filtered[1]))) / (
                        self.filtered[0] * self.filtered[1])).to(self.device)
        return x

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        x1_in = torch.softmax(self.convolve_in(r_idx, e1_idx), dim=1)
        x2_in = torch.softmax(self.convolve_in(r_idx, e2_idx), dim=1)
        x3_in = torch.softmax(self.convolve_in(r_idx, e3_idx), dim=1)
        x4_in = torch.softmax(self.convolve_in(r_idx, e4_idx), dim=1)
        x5_in = torch.softmax(self.convolve_in(r_idx, e5_idx), dim=1)
        x6_in = torch.softmax(self.convolve_in(r_idx, e6_idx), dim=1)
        x_in = x1_in * x2_in * x3_in * x4_in * x5_in * x6_in
        # print(self.p.device)
        # print(self.hidden_drop(x_in).device)
        x = torch.mm(self.hidden_drop(x_in), self.p)

        # x = torch.sum(self.hidden_drop(x_in), dim=1)

        return x


class InteractE(BaseClass):
    def __init__(self, logger, dataset, max_arity, **kwargs):
        super(InteractE, self).__init__()
        self.device = kwargs['device']
        self.logger = logger
        self.has_run_debug = True
        self.max_arity = max_arity
        self.inp_drop_rate = kwargs['inp_drop']
        self.feature_map_drop_rate = kwargs['feature_map_drop']
        self.hidden_drop_rate = kwargs['hidden_drop']

        self.emb_dim = kwargs['emb_dim']
        self.E = torch.nn.Embedding(dataset.num_ent(), self.emb_dim, padding_idx=0)
        self.rel_num = kwargs['R_external_dim']
        self.R = torch.nn.Embedding(dataset.num_rel(), self.rel_num, padding_idx=0)

        self.chequer_perm = kwargs['chequer_perm']  # ex 排列具体详情
        self.perm = kwargs['perm']
        self.k_w = kwargs['k_w']
        self.k_h = kwargs['k_h']
        self.ker_sz = kwargs['ker_sz']
        self.num_ft = kwargs['num_ft']
        self.padding = 0
        self.register_parameter('conv_filt', Parameter(torch.zeros(self.num_ft, 1, self.ker_sz, self.ker_sz)))
        xavier_normal_(self.conv_filt)
        self.flat_sz = self.num_ft * (self.k_w * 2 + 1) * (self.k_h + 1) * self.perm

        self.bn3 = torch.nn.BatchNorm2d(self.perm)
        self.bn4 = torch.nn.BatchNorm2d(self.perm * self.num_ft)

        self.input_drop = torch.nn.Dropout(self.inp_drop_rate)
        self.feature_map_drop = torch.nn.Dropout(self.feature_map_drop_rate)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

        self.p = torch.rand(((self.k_w * 2 + 1) * (self.k_h + 1), 1)).to(self.device)
        self.init()
        self.logger.debug(f'self.E.weight.data[0]={self.E.weight.data[0]}')

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.rel_num)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])

    def convolve_ex(self, r_idx, e_idx):
        r = self.R(r_idx)
        e = self.E(e_idx)
        batch_size = len(e_idx)
        comb_emb = torch.cat([e, r], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        # print(self.chequer_perm.size())
        stack_inp = chequer_perm.reshape((-1, self.perm, 2 * self.k_w, self.k_h))
        stack_inp = self.bn3(stack_inp)
        x = self.input_drop(stack_inp)
        x = circular_padding_chw(x, self.ker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.perm, 1, 1, 1), padding=self.padding, groups=self.perm)
        # print(x.size())
        x = self.bn4(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1, self.k_w * 2 + 1, self.k_h + 1)
        x = x.sum(dim=1)
        x = x.view(batch_size, -1)
        for idd, i in enumerate(e_idx):
            if i == 0:
                x[idd] = torch.tensor(np.ones((1, (self.k_w * 2 + 1) * (self.k_h + 1)))) / (
                            (self.k_w * 2 + 1) * (self.k_h + 1)).to(self.device)
        return x

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        x1_in = torch.softmax(self.convolve_ex(r_idx, e1_idx), dim=1)
        x2_in = torch.softmax(self.convolve_ex(r_idx, e2_idx), dim=1)
        x3_in = torch.softmax(self.convolve_ex(r_idx, e3_idx), dim=1)
        x4_in = torch.softmax(self.convolve_ex(r_idx, e4_idx), dim=1)
        x5_in = torch.softmax(self.convolve_ex(r_idx, e5_idx), dim=1)
        x6_in = torch.softmax(self.convolve_ex(r_idx, e6_idx), dim=1)
        x_in = x1_in * x2_in * x3_in * x4_in * x5_in * x6_in
        x = torch.mm(self.hidden_drop(x_in), self.p)

        # x = torch.sum(self.hidden_drop(x_in), dim=1)

        return x


class GRACE(BaseClass):
    def __init__(self, logger, dataset, max_arity, **kwargs):
        super(GRACE, self).__init__()
        self.device = kwargs['device']
        self.logger = logger
        self.has_run_debug = True
        self.max_arity = max_arity
        self.inp_drop_rate = kwargs['inp_drop']
        self.feature_map_drop_rate = kwargs['feature_map_drop']
        self.hidden_drop_rate = kwargs['hidden_drop']

        self.emb_dim = kwargs['emb_dim']
        self.reshape = kwargs['reshape']
        self.E = torch.nn.Embedding(dataset.num_ent(), self.emb_dim, padding_idx=0).to(self.device)
        self.E1 = torch.nn.Embedding(dataset.num_ent(), self.emb_dim, padding_idx=0).to(self.device)

        self.rel_num = kwargs['R_internal_dim']
        self.rel_num1 = kwargs['R_external_dim']
        self.kernel_size = kwargs['kernel_size']
        self.R_num = int(self.rel_num / (self.kernel_size[0] * self.kernel_size[1]))
        self.logger.debug(f'self.R_num={self.R_num}')
        self.R = torch.nn.Embedding(dataset.num_rel(), self.rel_num, padding_idx=0)
        self.R1 = torch.nn.Embedding(dataset.num_rel(), self.rel_num1, padding_idx=0)

        self.chequer_perm = kwargs['chequer_perm']
        self.perm = kwargs['perm']
        self.k_w = kwargs['k_w']
        self.k_h = kwargs['k_h']
        self.ker_sz = kwargs['ker_sz']
        self.num_ft = kwargs['num_ft']
        self.padding = 0
        self.register_parameter('conv_filt', Parameter(torch.zeros(self.num_ft, 1, self.ker_sz, self.ker_sz)))
        xavier_normal_(self.conv_filt)
        self.flat_sz = self.num_ft * (self.k_w * 2 + 1) * (self.k_h + 1) * self.perm

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.R_num)
        self.bn2 = torch.nn.BatchNorm1d(self.rel_num)
        self.bn3 = torch.nn.BatchNorm2d(self.perm)
        self.bn4 = torch.nn.BatchNorm2d(self.perm * self.num_ft)

        self.input_drop = torch.nn.Dropout(self.inp_drop_rate)
        self.feature_map_drop = torch.nn.Dropout(self.feature_map_drop_rate)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

        self.filtered = [self.reshape[0] - self.kernel_size[0] + 1, self.reshape[1] - self.kernel_size[1] + 1]
        self.p1 = torch.rand(((self.k_w * 2 + 1) * (self.k_h + 1), 1)).to(self.device)
        self.p = torch.rand((self.filtered[0] * self.filtered[1], 1)).to(self.device)
        self.init()
        self.logger.debug(f'self.E.weight.data[0]={self.E.weight.data[0]}')



    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.rel_num)
        self.E1.weight.data[0] = torch.ones(self.emb_dim)
        self.R1.weight.data[0] = torch.ones(self.rel_num1)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])
        xavier_uniform_(self.E1.weight.data[1:])
        xavier_uniform_(self.R1.weight.data[1:])

    def convolve_ex(self, r_idx, e_idx):
        r = self.R1(r_idx)
        e = self.E1(e_idx)
        batch_size = len(e_idx)
        comb_emb = torch.cat([e, r], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        # print(self.chequer_perm.size())
        stack_inp = chequer_perm.reshape((-1, self.perm, 2 * self.k_w, self.k_h))
        stack_inp = self.bn3(stack_inp)
        x = self.input_drop(stack_inp)
        x = circular_padding_chw(x, self.ker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.perm, 1, 1, 1), padding=self.padding, groups=self.perm)
        # print(x.size())
        x = self.bn4(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1, self.k_w * 2 + 1, self.k_h + 1)
        x = x.sum(dim=1)
        x = x.view(batch_size, -1)
        for idd, i in enumerate(e_idx):
            if i == 0:
                x[idd] = (torch.tensor(np.ones((1, (self.k_w * 2 + 1) * (self.k_h + 1)))) / (
                            (self.k_w * 2 + 1) * (self.k_h + 1))).to(self.device)
        return x

    def convolve_in(self, r_idx, e_idx):
        batch_size = len(e_idx)
        e = self.E(e_idx).view(batch_size, 1, *self.reshape)
        self.logger.debug(f'The embedded dimension of the entity matrix is {e.size()}') if self.has_run_debug else 1
        e = self.bn0(e).view(1, batch_size, *self.reshape)
        e = self.input_drop(e)

        r = self.R(r_idx)
        r = self.bn2(r)
        r = self.input_drop(r)
        r = r.view(self.R_num * batch_size, 1, *self.kernel_size)

        self.logger.debug(f'The embedded dimension of the relation matrix is {r.size()}') if self.has_run_debug else 1
        self.has_run_debug = False

        x = F.conv2d(e, r, groups=batch_size)
        x = x.view(batch_size, self.R_num, *self.filtered)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.sum(dim=1)
        x = x.view(batch_size, -1)
        for idd, i in enumerate(e_idx):
            if i == 0:
                x[idd] = (torch.tensor(np.ones((1, self.filtered[0] * self.filtered[1]))) / (
                        self.filtered[0] * self.filtered[1])).to(self.device)
        return x

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        x1_in = torch.softmax(self.convolve_in(r_idx, e1_idx), dim=1)
        x2_in = torch.softmax(self.convolve_in(r_idx, e2_idx), dim=1)
        x3_in = torch.softmax(self.convolve_in(r_idx, e3_idx), dim=1)
        x4_in = torch.softmax(self.convolve_in(r_idx, e4_idx), dim=1)
        x5_in = torch.softmax(self.convolve_in(r_idx, e5_idx), dim=1)
        x6_in = torch.softmax(self.convolve_in(r_idx, e6_idx), dim=1)
        x_in = x1_in * x2_in * x3_in * x4_in * x5_in * x6_in
        x = torch.mm(self.hidden_drop(x_in), self.p)

        x1_ex = torch.softmax(self.convolve_ex(r_idx, e1_idx), dim=1)
        x2_ex = torch.softmax(self.convolve_ex(r_idx, e2_idx), dim=1)
        x3_ex = torch.softmax(self.convolve_ex(r_idx, e3_idx), dim=1)
        x4_ex = torch.softmax(self.convolve_ex(r_idx, e4_idx), dim=1)
        x5_ex = torch.softmax(self.convolve_ex(r_idx, e5_idx), dim=1)
        x6_ex = torch.softmax(self.convolve_ex(r_idx, e6_idx), dim=1)
        x_ex = x1_ex * x2_ex * x3_ex * x4_ex * x5_ex * x6_ex
        x = x + torch.mm(self.hidden_drop(x_ex), self.p1)

        # x = torch.sum(self.hidden_drop(x_in), dim=1)

        return x


class HyperConvD(BaseClass):
    def __init__(self, logger, dataset, max_arity, **kwargs):
        super(HyperConvD, self).__init__()
        self.device = kwargs['device']
        self.logger = logger
        self.has_run_debug = True
        self.max_arity = max_arity
        self.inp_drop_rate = kwargs['inp_drop']
        self.feature_map_drop_rate = kwargs['feature_map_drop']
        self.hidden_drop_rate = kwargs['hidden_drop']

        self.emb_dim = kwargs['emb_dim']
        self.reshape = kwargs['reshape']
        self.E = torch.nn.Embedding(dataset.num_ent(), self.emb_dim, padding_idx=0).to(self.device)

        self.rel_num = kwargs['R_internal_dim']
        self.kernel_size = kwargs['kernel_size']
        self.R_num = int(self.rel_num / (self.kernel_size[0] * self.kernel_size[1]))
        self.logger.debug(f'self.R_num={self.R_num}')
        self.R = torch.nn.Embedding(dataset.num_rel(), self.rel_num, padding_idx=0).to(self.device)

        self.bn0 = torch.nn.BatchNorm2d(1).to(self.device)
        self.bn1 = torch.nn.BatchNorm2d(self.R_num).to(self.device)
        self.bn2 = torch.nn.BatchNorm1d(self.rel_num).to(self.device)

        self.input_drop = torch.nn.Dropout(self.inp_drop_rate).to(self.device)
        self.feature_map_drop = torch.nn.Dropout(self.feature_map_drop_rate).to(self.device)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate).to(self.device)

        self.filtered = [self.reshape[0] - self.kernel_size[0] + 1, self.reshape[1] - self.kernel_size[1] + 1]
        self.p = torch.rand((100, 1)).to(self.device)
        self.init()
        self.logger.debug(f'self.E.weight.data[0]={self.E.weight.data[0]}')

        self.fc_2 = nn.Linear(in_features=(2 * self.reshape[0] - 2) * self.filtered[1], out_features=100).to(self.device)
        self.fc_3 = nn.Linear(in_features=(3 * self.reshape[0] - 2) * self.filtered[1], out_features=100).to(self.device)
        self.fc_4 = nn.Linear(in_features=(4 * self.reshape[0] - 2) * self.filtered[1], out_features=100).to(self.device)
        self.fc_5 = nn.Linear(in_features=(5 * self.reshape[0] - 2) * self.filtered[1], out_features=100).to(self.device)
        self.fc_6 = nn.Linear(in_features=(6 * self.reshape[0] - 2) * self.filtered[1], out_features=100).to(self.device)

        self.Q = nn.Parameter(torch.randn((self.emb_dim, self.emb_dim))).to(self.device)
        self.K = nn.Parameter(torch.randn((self.rel_num, self.emb_dim))).to(self.device)
        self.V = nn.Parameter(torch.randn((self.rel_num, self.emb_dim))).to(self.device)

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim).to(self.device)
        self.R.weight.data[0] = torch.ones(self.rel_num).to(self.device)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])

    def convolve_in(self, r_idx, e_idx):
        batch_size = len(e_idx)
        e = self.E(e_idx).view(batch_size, 1, *self.reshape)
        self.logger.debug(f'The embedded dimension of the entity matrix is {e.size()}') if self.has_run_debug else 1
        e = self.bn0(e).view(1, batch_size, *self.reshape)
        e = self.input_drop(e)

        r = self.R(r_idx)
        r = self.bn2(r)
        r = self.input_drop(r)
        r = r.view(self.R_num * batch_size, 1, *self.kernel_size)

        self.logger.debug(f'The embedded dimension of the relation matrix is {r.size()}') if self.has_run_debug else 1
        self.has_run_debug = False

        x = F.conv2d(e, r, groups=batch_size)
        x = x.view(batch_size, self.R_num, *self.filtered)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.sum(dim=1)
        x = x.view(batch_size, -1)
        # x = torch.softmax(x, dim=1)
        for idd, i in enumerate(e_idx):
            if i == 0:
                x[idd] = torch.tensor(torch.tensor(np.ones((1, self.filtered[0] * self.filtered[1]))) / (
                        self.filtered[0] * self.filtered[1])).to(self.device)
        return x

    def attention(self, e, r):
        Q = torch.mm(e, self.Q)
        K = torch.mm(r, self.K)
        # v = self.V[:e.size(0), :]
        V = torch.mm(r, self.V)

        res = torch.mm(Q, K.T) / sqrt(e.size(1))
        res = torch.softmax(res, dim=1)
        attention = torch.mm(res, V)
        return attention
    def Adaptive(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        batch_size = len(e1_idx)
        e1 = self.bn0(self.E(e1_idx).view(batch_size, 1, self.emb_dim, 1))
        e2 = self.bn0(self.E(e2_idx).view(batch_size, 1, self.emb_dim, 1))
        e3 = self.bn0(self.E(e3_idx).view(batch_size, 1, self.emb_dim, 1))
        e4 = self.bn0(self.E(e4_idx).view(batch_size, 1, self.emb_dim, 1))
        e5 = self.bn0(self.E(e5_idx).view(batch_size, 1, self.emb_dim, 1))
        e6 = self.bn0(self.E(e6_idx).view(batch_size, 1, self.emb_dim, 1))
        if e3_idx[0] == 0:
            x = torch.cat((e1, e2), dim=1)
            arr = 2
        elif e4_idx[0] == 0:
            x = torch.cat((e1, e2, e3), dim=1)
            arr = 3
        elif e6_idx[0] == 0:
            x = torch.cat((e1, e2, e3, e4), dim=1)
            arr = 4
        elif e6_idx[0] == 0:
            x = torch.cat((e1, e2, e3, e4, e5), dim=1)
            arr = 5
        else:
            x = torch.cat((e1, e2, e3, e4, e5, e6), dim=1)
            arr = 6
        e = torch.mean(x, dim=1)
        e = e.view(batch_size, -1)
        x = self.input_drop(x)
        r = self.R(r_idx)
        attention = self.attention(e, r)
        r = self.bn2(r)
        r = self.input_drop(r)
        r = r.view(self.R_num * batch_size, 1, *self.kernel_size)
        x = x.view(1, batch_size, self.reshape[0]*arr, self.reshape[1])
        x = F.conv2d(x, r, groups=batch_size)
        self.filtered[0] = 10 * arr -2
        x = x.view(batch_size, self.R_num, self.filtered[0], self.filtered[1])
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = attention.view(batch_size, self.R_num, 1, 1) * x
        x = x.sum(dim=1)
        if arr == 2:
            x = self.fc_2(x.view(batch_size, -1))
        elif arr == 3:
            x = self.fc_3(x.view(batch_size, -1))
        elif arr == 4:
            x = self.fc_4(x.view(batch_size, -1))
        elif arr == 5:
            x = self.fc_5(x.view(batch_size, -1))
        else:
            x = self.fc_6(x.view(batch_size, -1))
        x = self.hidden_drop(x)
        return x

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        x = self.Adaptive(r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx)
        scores = torch.mm(x, self.p)


        return scores
