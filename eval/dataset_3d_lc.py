import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import csv
import glob
import pandas as pd
import numpy as np
import cv2
sys.path.append('../utils')
from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq =1,
                 downsample=3,
                 epsilon=5,
                 which_split=1):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split

        # splits
        if mode == 'train':
            split = '../process_data/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '../process_data/data/ucf101/test_split%02d.csv' % self.which_split # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join('../process_data/data/ucf101', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
        return [seq_idx_block, vpath]


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform

        num_crop = None
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            num_crop = 5
            t_seq = torch.stack(tmp, 1)

        if self.mode == 'test':
            # return all available clips, but cut into length = num_seq
            SL = t_seq.size(0)
            clips = []; i = 0
            while i+self.seq_len <= SL:
                clips.append(t_seq[i:i+self.seq_len, :])
                # i += self.seq_len//2
                i += self.seq_len
            if num_crop:
                # half overlap:
                clips = [torch.stack(clips[i:i+self.num_seq], 0).permute(2,0,3,1,4,5) for i in range(0,len(clips)+1-self.num_seq,self.num_seq//2)]
                NC = len(clips)
                t_seq = torch.stack(clips, 0).view(NC*num_crop, self.num_seq, C, self.seq_len, H, W)
            else:
                # half overlap:
                clips = [torch.stack(clips[i:i+self.num_seq], 0).transpose(1,2) for i in range(0,len(clips)+1-self.num_seq,self.num_seq//2)]
                t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        try:
            vname = vpath.split('/')[-3]
            vid = self.encode_action(vname)
        except:
            vname = vpath.split('/')[-2]
            vid = self.encode_action(vname)

        label = torch.LongTensor([vid])

        return t_seq, label

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class HMDB51_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=1,
                 downsample=1,
                 epsilon=5,
                 which_split=1):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split

        # splits
        if mode == 'train':
            split = '../process_data/data/hmdb51/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '../process_data/data/hmdb51/test_split%02d.csv' % self.which_split # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join('../process_data/data/hmdb51', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
        return [seq_idx_block, vpath]


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform

        num_crop = None
        try:
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
        except:
            (C, H, W) = t_seq[0][0].size()
            tmp = [torch.stack(i, 0) for i in t_seq]
            assert len(tmp) == 5
            num_crop = 5
            t_seq = torch.stack(tmp, 1)
        # print(t_seq.size())
        # import ipdb; ipdb.set_trace()
        if self.mode == 'test':
            # return all available clips, but cut into length = num_seq
            SL = t_seq.size(0)
            clips = []; i = 0
            while i+self.seq_len <= SL:
                clips.append(t_seq[i:i+self.seq_len, :])
                # i += self.seq_len//2
                i += self.seq_len
            if num_crop:
                # half overlap:
                clips = [torch.stack(clips[i:i+self.num_seq], 0).permute(2,0,3,1,4,5) for i in range(0,len(clips)+1-self.num_seq,self.num_seq//2)]
                NC = len(clips)
                t_seq = torch.stack(clips, 0).view(NC*num_crop, self.num_seq, C, self.seq_len, H, W)
            else:
                # half overlap:
                clips = [torch.stack(clips[i:i+self.num_seq], 0).transpose(1,2) for i in range(0,len(clips)+1-self.num_seq,3*self.num_seq//4)]
                t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        try:
            vname = vpath.split('/')[-3]
            vid = self.encode_action(vname)
        except:
            vname = vpath.split('/')[-2]
            vid = self.encode_action(vname)

        label = torch.LongTensor([vid])

        return t_seq, label

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]

