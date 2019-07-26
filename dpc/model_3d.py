import time
import math
import numpy as np
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')
from select_backbone import select_resnet
from convrnn import ConvGRU


class DPC_RNN(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3,
                 network='resnet50', version='v1', full_resnet=True, network_type='2d3d', num_fc=1,
                 batchnorm='bn', pooled_size=6, affine=True, seq_pred=True):
        super(DPC_RNN, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.batchnorm = batchnorm
        self.num_fc = num_fc
        self.seq_pred = seq_pred
        self.full_resnet = full_resnet
        self.version = version
        self.pooled_size = pooled_size
        self.last_duration = int(math.ceil(seq_len / 4))

        if self.last_duration <= 0: raise ValueError

        if batchnorm: print('Use: %s' % batchnorm)
        print('feature map pooled into %dx%d' % (pooled_size, pooled_size))

        self.last_size = int(math.ceil(sample_size / 32))
        print('=> Using full resnet %s' % self.version)

        self.resnet, self.param = select_resnet(network, network_type, sample_size,
                                                seq_len, batchnorm, affine,
                                                version=version,
                                                full_resnet=full_resnet,
                                                track_running_stats=False)

        self.param['num_layers'] = 1
        self.param['hidden_size'] = self.param['feature_size']

        assert self.last_size % pooled_size == 0
        self.param['pool_kernel_size'] = int(self.last_size / pooled_size)
        self.last_size = pooled_size

        self.convrnn = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])

        # 1 FC version
        if num_fc == 1:
            self.network_pred = nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
        elif num_fc == 2:
            self.network_pred = nn.Sequential(
                                        nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                        )

        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.convrnn)
        self._initialize_weights(self.network_pred)

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.resnet(block)
        del block
        feature = F.avg_pool3d(feature,
                               (self.last_duration, self.param['pool_kernel_size'], self.param['pool_kernel_size']),
                               stride=(1, self.param['pool_kernel_size'], self.param['pool_kernel_size'])) # only pool on temporal axis, as feature is short on T

        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)
        feature = self.relu(feature) # [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B,N,d=1024,6,6] [0, +inf)
        feature_inf = feature_inf_all[:, N-self.pred_step::, :].contiguous()
        del feature_inf_all

        _, hidden = self.convrnn(feature[:, 0:N-self.pred_step, :].contiguous())
        hidden = hidden[:,-1,:] # after tanh, (-1,1) hidden state of last layer, last time step

        pred = []
        for i in range(self.pred_step):
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.convrnn(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:,-1,:]
        pred = torch.stack(pred, 1) # B, pred_step, xxx
        del hidden

        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]
        N = self.pred_step
        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT. 
        pred = pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*self.last_size**2, self.param['feature_size'])
        feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*self.last_size**2, self.param['feature_size']).transpose(0,1)
        score = torch.matmul(pred, feature_inf).view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
        del feature_inf, pred

        if self.mask is None: # only compute mask once at the beginning
            # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
            mask = torch.zeros((B, self.pred_step, self.last_size**2, B, N, self.last_size**2), dtype=torch.int8, requires_grad=False).detach().cuda()
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3 # spatial neg
            for k in range(B):
                mask[k, :, torch.arange(self.last_size**2), k, :, torch.arange(self.last_size**2)] = -1 # temporal neg
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B*self.last_size**2, self.pred_step, B*self.last_size**2, N)
            for j in range(B*self.last_size**2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(N-self.pred_step, N)] = 1 # pos
            mask = tmp.view(B, self.last_size**2, self.pred_step, B, self.last_size**2, N).permute(0,2,1,3,5,4)
            self.mask = mask

        return [score, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1) # default gain is 1
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None

