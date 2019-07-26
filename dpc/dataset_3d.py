import torch
from torch.utils import data
from torchvision import transforms
import glob
import os
import sys
import csv
import pandas as pd
import numpy as np
import cv2
sys.path.append('../utils')
from augmentation import *
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import time

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def read_file(path,):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content


class Kinetics400_full_3d(data.Dataset):
    def __init__(self,
                 root='/scratch/local/ssd/htd/kinetics400/',
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 num_proposal=None,
                 unit_test=False,
                 big=False,
                 return_label=False):
        self.root = root # seems not useful
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.num_proposal = num_proposal
        self.unit_test = unit_test
        self.return_label = return_label

        if big: print('Using Kinetics400 full data (256x256)')
        else: print('Using Kinetics400 full data (150x150)')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        if not os.path.exists(os.path.join(self.root, '..','ClassInd.txt')):
            self.root = '/scratch/shared/nfs1/htd/kinetics400' # triton
        action_file = os.path.join(self.root, 'ClassInd.txt')
        action_df = pd.read_csv(action_file, sep=',', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # splits
        if big:
            if mode == 'train':
                split = '/users/htd/Desktop/SelfSupervision/process_data/data/kinetics400_256/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '/users/htd/Desktop/SelfSupervision/process_data/data/kinetics400_256/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')
        else: # small
            if mode == 'train':
                split = '/users/htd/Desktop/SelfSupervision/process_data/data/kinetics400/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '/users/htd/Desktop/SelfSupervision/process_data/data/kinetics400/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        if self.unit_test: self.video_info = self.video_info.sample(32, random_state=666)
        # shuffle not necessary because use RandomSampler

    def sub_sampler(self, n, vlen, vpath):
        # only works for n=1 now
        result = []
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        if self.num_proposal is not None: assert self.num_proposal > n
        n_ = self.num_proposal if self.num_proposal else n
        start_idx_pool = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n_)
        seq_idx_pool = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx_pool

        if self.num_proposal is not None:
            seq_idx_pool = self.sampling_start_idx_with_proposal(seq_idx_pool, vpath, n)

        for idx in range(n):
            seq_idx_block = seq_idx_pool + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
            result.append([seq_idx_block,
                           vpath, # tell which video
                          ])
        return result

    def sampling_start_idx_with_proposal(self, seq_idx_pool, vpath, n, p=0.0):
        test_idx_pool = seq_idx_pool
        idx_pool = np.random.choice([i for i in range(test_idx_pool.shape[-1])], n)
        return seq_idx_pool[:, idx_pool]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.sub_sampler(1, vlen, vpath)[0] # idx 0 because only select 1 sequence from 1 video
        if items is None: print(vpath) 
        
        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq*self.seq_len)
        
        # seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        seq = Parallel(n_jobs=1, prefer='threads')\
               (delayed(pil_loader)(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block) # multiple imgs, a list; jpg if necessary

        t_seq = self.transform(seq) # apply same transform
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)

            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class UCF101_3d(data.Dataset):
    def __init__(self,
                 root='/scratch/local/ssd/datasets/UCF101',
                 mode='train',
                 transform=None, 
                 seq_len=10,
                 num_seq = 5,
                 downsample=3,
                 epsilon=5,
                 num_proposal=3,
                 which_split=1,
                 return_label=False):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.num_proposal = num_proposal
        self.which_split = which_split
        self.return_label = return_label

        # splits
        if mode == 'train':
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'): # use val for test
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/ucf101/test_split%02d.csv' % self.which_split 
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        if not os.path.exists(os.path.join(self.root, 'splits_classification')):
            self.root = '/scratch/shared/nfs1/htd/UCF101' # triton
        action_file = os.path.join(self.root, 'splits_classification', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
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
        # shuffle not required due to external sampler

    def sub_sampler(self, n, vlen, vpath):
        # only works for n=1 now
        result = []
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        if self.num_proposal is not None: assert self.num_proposal > n
        n_ = self.num_proposal if self.num_proposal else n
        start_idx_pool = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n_)
        seq_idx_pool = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx_pool

        seq_idx_pool = self.sampling_start_idx_with_proposal(seq_idx_pool, vpath, n)

        for idx in range(n):
            seq_idx_block = seq_idx_pool + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
            result.append([seq_idx_block, vpath])
        return result

    def sampling_start_idx_with_proposal(self, seq_idx_pool, vpath, n):
        test_idx_pool = seq_idx_pool
        idx_pool = np.random.choice([i for i in range(test_idx_pool.shape[-1])], n)
        return seq_idx_pool[:, idx_pool]


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.sub_sampler(1, vlen, vpath)[0] # idx 0 because only select 1 sequence from 1 video
        if items is None: print(vpath) 
        
        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq*self.seq_len)
        
        # seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        seq = Parallel(n_jobs=1, prefer='threads')\
               (delayed(pil_loader)(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block) # multiple imgs, a list; jpg if necessary
        
        t_seq = self.transform(seq) # apply same transform
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq, label
            
        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]

