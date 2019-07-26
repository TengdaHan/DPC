import torch.nn as nn
import torch
import math
import numpy as np
# resnet backbone
import sys
sys.path.append('../../backbone')
from select_backbone import select_resnet
from convrnn import ConvGRU

import torch.nn.functional as F

class LC(nn.Module):
    def __init__(self, sample_size, num_seq, seq_len, 
                 network='resnet18', version='v1', full_resnet=True, network_type='2d3d', dropout=0.5,
                 num_class=101, batchnorm=False, num_fc=1, norm=False, use_rnn=False, rnn_kernel=3, use_final_bn=True):
        super(LC, self).__init__()
        torch.cuda.manual_seed(666)
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.num_class = num_class 
        
        self.batchnorm = batchnorm 
        self.norm = norm 
        self.use_rnn = use_rnn
        self.rnn_kernel = rnn_kernel 
        self.use_final_bn = True # manually set  
        self.full_resnet = full_resnet
        self.version = version

        print('=> Using RNN + FC model ')

        if network_type == '3d':
            print('=> Use 3D %s' % network)
            if self.version == 'v1':
                if full_resnet:
                    self.last_duration = int(math.ceil(seq_len / 16))
                else:
                    self.last_duration = int(math.ceil(seq_len / 8))
            elif self.version == 'v2':
                if full_resnet:
                    self.last_duration = int(seq_len-8)
                else:
                    self.last_duration = int(seq_len-6)
        else:
            print('=> Use 2D-3D %s!' % network)
            if self.version == 'v1':
                if full_resnet:
                    self.last_duration = int(math.ceil(seq_len / 4))
                else:
                    self.last_duration = int(math.ceil(seq_len / 2))
            elif self.version == 'v2':
                if full_resnet:
                    self.last_duration = int(seq_len-4)
                else:
                    self.last_duration = int(seq_len-2)
        if self.last_duration <= 0: raise ValueError

        if full_resnet:
            self.last_size = int(math.ceil(sample_size / 32))
            print('=> Using full resnet')
        else:
            self.last_size = int(math.ceil(sample_size / 16))
            print('=> Using truncated resnet')

        if (network=='resnet18') or (network=='resnet34'):
            track_running_stats = True # small network, track running stat
        else:
            track_running_stats = False # big network, due to small batchsize, use batch stat

        self.resnet, self.param = select_resnet(network, network_type, sample_size, 
                                                seq_len, batchnorm, 
                                                version=version, 
                                                full_resnet=full_resnet,
                                                track_running_stats=track_running_stats)
        self.param['num_layers'] = 1

        self.param['hidden_size'] = self.param['feature_size'] # 256 or 1024
        # if network == 'resnet50':
        #     self.param['hidden_size'] = 256
        # elif (network == 'resnet18') or (network == 'resnet34'):
        #     self.param['hidden_size'] = self.param['feature_size']*2

        if use_rnn:
            if rnn_kernel == 3:
                print('=> using ConvRNN, kernel_size=3')
                self.final_rnn = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=3,
                               num_layers=self.param['num_layers'])
                self._initialize_weights(self.final_rnn)
                self.final_context = nn.Conv2d(self.param['hidden_size'], self.param['feature_size'], kernel_size=1, padding=0)
                self._initialize_weights(self.final_context)
            elif rnn_kernel == 1:
                print('=> using ConvRNN, kernel_size=1')
                self.convrnn = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])
                self._initialize_weights(self.convrnn)
                self.network_context = nn.Conv2d(self.param['hidden_size'], self.param['feature_size'], kernel_size=1, padding=0)
                self._initialize_weights(self.network_context)
            else:
                print('=> using normal RNN')
                self.final_rnn = nn.GRU(input_size=self.param['feature_size'],
                                  hidden_size=self.param['feature_size'],
                                  num_layers=self.param['num_layers'],
                                  batch_first=True)
                self._initialize_weights(self.final_rnn)
                self.final_context = nn.Linear(self.param['feature_size'], self.param['feature_size'])
                self._initialize_weights(self.final_context)
        else:
            print('=> no RNN')

        if self.use_final_bn: 
            print('=> always using final BN')
            self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()

        if num_fc == 1:
            self.final_fc = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(self.param['feature_size'], self.num_class)
                            )
        elif num_fc == 2:
            self.final_fc = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(self.param['feature_size'], self.param['feature_size']),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.param['feature_size'], self.num_class)
                            )
        elif num_fc == 3:
            self.final_fc = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(self.param['feature_size'], self.param['feature_size']),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(self.param['feature_size'], self.param['feature_size']),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.param['feature_size'], self.num_class)
                            )
        
        self._initialize_weights(self.final_fc)

    def forward(self, block, h0):
        # seq1: [B, N, C, SL, W, H]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.resnet(block)
        del block 
        # if self.batchnorm: feature = self.final_bn(feature)
        feature = F.relu(feature)
        
        if self.use_rnn:
            # TODO: potential issue here?
            if self.rnn_kernel == 3: # obsolete now
                raise NotImplementedError()
                feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
                feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B*N,d=1024,6,6]
                context, _ = self.final_rnn(feature)
                # context = context[:,-1,:].unsqueeze(1)
                context = F.relu(self.final_context(context[:,-1,:]).unsqueeze(1))
                context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
            elif self.rnn_kernel == 1:
                feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
                feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B*N,d=1024,6,6]
                context, _ = self.convrnn(feature)
                # context = context[:,-1,:].unsqueeze(1)
                context = F.relu(self.network_context(context[:,-1,:]).unsqueeze(1))
                # context = F.relu(self.network_context(context.view(B*N,self.param['feature_size'],self.last_size,self.last_size))).view(B,N,self.param['feature_size'],self.last_size,self.last_size)
                context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
            else:
                feature = F.avg_pool3d(feature, (self.last_duration, self.last_size, self.last_size), stride=1)
                feature = feature.view(B, N, self.param['feature_size']) # [B*N,d=1024]
                h0 = h0.transpose(0,1).contiguous()
                self.final_rnn.flatten_parameters()
                context, _ = self.final_rnn(feature, h0) # [B, N, d]
                # context = context[:,-1,:].unsqueeze(1)
                context = F.relu(self.final_context(context[:,-1,:]).unsqueeze(1))
                # context = F.relu(self.final_context(context))
        else:
            feature = F.avg_pool3d(feature, (self.last_duration, self.last_size, self.last_size), stride=1)
            feature = feature.view(B, N, self.param['feature_size']) # [B*N,d=1024]
            context = feature
        del feature

        # for linear classifier, either N=1(train/val) or B=1 (test), therefore:
        # if self.norm: context = context / context.norm(dim=2, keepdim=True) # this line is not useful
        if self.use_final_bn: 
            context = self.final_bn(context.transpose(-1,-2)).transpose(-1,-2) # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
        output = self.final_fc(context).view(B, -1, self.num_class)

        return output, context

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                # nn.init.xavier_normal_(param)
                nn.init.orthogonal_(param, 1) # default gain is 1           
        # other resnet weights have been initialized in resnet_3d.py

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, # batch first here, for convenient data parallel
                           self.param['num_layers'],
                           self.param['feature_size'])


