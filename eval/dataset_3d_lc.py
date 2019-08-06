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

class Kinetics400_3d(data.Dataset):
    '''Kinetics Subset. Obsolete!'''
    def __init__(self,
                 root='/scratch/local/ssd/htd/kinetics400/frame_subset',
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq = 1,
                 downsample=3,
                 epsilon=5,
                 num_proposal=3,
                 unit_test=False,
                 test_type='c'):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.num_proposal = num_proposal
        self.unit_test = unit_test
        self.test_type = test_type

        # splits
        if mode == 'train':
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/kinetics400_subset/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/kinetics400_subset/val_split.csv'
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

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
            if act_id >= 40: break
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)
        # if mode == 'val': self.video_info = self.video_info.sample(frac=0.5)
        if self.unit_test: self.video_info = self.video_info.sample(4, random_state=666)

        # shuffle not necessary because use RandomSampler

    def sub_sampler(self, n, vlen, vpath):
        # only works for n=1 now
        result = []
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample) # all possible frames with downsampling
            return [[seq_idx_block, vpath]]
        if self.num_proposal is not None: assert self.num_proposal > n
        n_ = self.num_proposal if self.num_proposal else n
        start_idx_pool = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n_)
        seq_idx_pool = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx_pool

        if self.num_proposal is not None:
            seq_idx_pool = self.sampling_start_idx_with_proposal(seq_idx_pool, vpath, n, p=0.0)

        for idx in range(n):
            seq_idx_block = seq_idx_pool + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
            result.append([seq_idx_block,
                           vpath, # tell which video
                          ])
        return result

    def calc_mean_flow(self, idx_list, vpath):
        mean_flow = []
        fpath = vpath.replace('frame_subset', 'flow_subset', 1)
        for idx in idx_list:
            flow_path = os.path.join(fpath, 'flow_%05d.jpg' % (idx+1))
            img = cv2.imread(flow_path)
            if img is None:
                continue
            mean_flow.append(np.mean(np.abs(img[:,:,0:2] - 127)))
        if mean_flow == []:
            return 0.0
        else:
            return np.mean(mean_flow)

    def sampling_start_idx_with_proposal(self, seq_idx_pool, vpath, n, p=0.0):
        test_idx_pool = seq_idx_pool
        if np.random.rand() < p:
            result = []
            for col_idx in range(test_idx_pool.shape[-1]):
                result.append(self.calc_mean_flow(test_idx_pool[:,col_idx], vpath))
            sort_idx = np.argsort(result)
            idx_pool = [i for i,j in enumerate(sort_idx) if j >= len(sort_idx)-n] # n seq with largest mean flow
        else: # have (1-p) prob to random choice
            idx_pool = np.random.choice([i for i in range(test_idx_pool.shape[-1])], n)
        return seq_idx_pool[:, idx_pool]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.sub_sampler(1, vlen, vpath)[0] # idx 0 because only select 1 sequence from 1 video
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        # seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        seq = Parallel(n_jobs=1, prefer='threads')\
               (delayed(pil_loader)(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block) # multiple imgs, a list; jpg if necessary

        t_seq = self.transform(seq) # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        if self.mode == 'test':
            if self.test_type == 'a':
                # type A: return all available frames
                t_seq = t_seq.view(1, -1, C, H, W).transpose(1,2)
            elif self.test_type == 'b':
                # type B: return all available clips together (variable num_seq)
                SL = t_seq.size(0)
                clips = []; i = 0
                while i+self.seq_len <= SL:
                    clips.append(t_seq[i:i+self.seq_len, :])
                    # i += self.seq_len//2
                    i += self.seq_len
                t_seq = torch.stack(clips, 0).transpose(1,2)
            elif self.test_type == 'c':
                # type C: return all available clips, but cut into length = num_seq
                SL = t_seq.size(0)
                clips = []; i = 0
                while i+self.seq_len <= SL:
                    clips.append(t_seq[i:i+self.seq_len, :])
                    i += self.seq_len
                clips = [torch.stack(clips[i:i+self.num_seq], 0).transpose(1,2) for i in range(len(clips)+1-self.num_seq)]
                print(clips.__len__(), clips[0].size())
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


class Kinetics400_full_3d(data.Dataset):
    def __init__(self,
                 root='/scratch/local/ssd/htd/kinetics400/',
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq = 1,
                 downsample=3,
                 epsilon=5,
                 num_proposal=3,
                 unit_test=False,
                 test_type='c'):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.num_proposal = num_proposal
        self.unit_test = unit_test
        self.test_type = test_type
        print('Using Kinetics400 full data')

        # splits
        if mode == 'train':
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/kinetics400/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/kinetics400/val_split.csv'
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

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

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
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
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample) # all possible frames with downsampling
            return [[seq_idx_block, vpath]]
        if self.num_proposal is not None: assert self.num_proposal > n
        n_ = self.num_proposal if self.num_proposal else n
        start_idx_pool = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n_)
        seq_idx_pool = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx_pool

        if self.num_proposal is not None:
            seq_idx_pool = self.sampling_start_idx_with_proposal(seq_idx_pool, vpath, n, p=0.0)

        for idx in range(n):
            seq_idx_block = seq_idx_pool + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
            result.append([seq_idx_block,
                           vpath, # tell which video
                          ])
        return result

    def calc_mean_flow(self, idx_list, vpath):
        mean_flow = []
        fpath = vpath.replace('frame_subset', 'flow_subset', 1)
        for idx in idx_list:
            flow_path = os.path.join(fpath, 'flow_%05d.jpg' % (idx+1))
            img = cv2.imread(flow_path)
            if img is None:
                continue
            mean_flow.append(np.mean(np.abs(img[:,:,0:2] - 127)))
        if mean_flow == []:
            return 0.0
        else:
            return np.mean(mean_flow)

    def sampling_start_idx_with_proposal(self, seq_idx_pool, vpath, n, p=0.0):
        test_idx_pool = seq_idx_pool
        if np.random.rand() < p:
            result = []
            for col_idx in range(test_idx_pool.shape[-1]):
                result.append(self.calc_mean_flow(test_idx_pool[:,col_idx], vpath))
            sort_idx = np.argsort(result)
            idx_pool = [i for i,j in enumerate(sort_idx) if j >= len(sort_idx)-n] # n seq with largest mean flow
        else: # have (1-p) prob to random choice
            idx_pool = np.random.choice([i for i in range(test_idx_pool.shape[-1])], n)
        return seq_idx_pool[:, idx_pool]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.sub_sampler(1, vlen, vpath)[0] # idx 0 because only select 1 sequence from 1 video
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        # seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        seq = Parallel(n_jobs=1, prefer='threads')\
               (delayed(pil_loader)(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block) # multiple imgs, a list; jpg if necessary

        t_seq = self.transform(seq) # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        if self.mode == 'test':
            if self.test_type == 'a':
                # type A: return all available frames
                t_seq = t_seq.view(1, -1, C, H, W).transpose(1,2)
            elif self.test_type == 'b':
                # type B: return all available clips together (variable num_seq)
                SL = t_seq.size(0)
                clips = []; i = 0
                while i+self.seq_len <= SL:
                    clips.append(t_seq[i:i+self.seq_len, :])
                    # i += self.seq_len//2
                    i += self.seq_len
                t_seq = torch.stack(clips, 0).transpose(1,2)
            elif self.test_type == 'c':
                # type C: return all available clips, but cut into length = num_seq
                SL = t_seq.size(0)
                clips = []; i = 0
                while i+self.seq_len <= SL:
                    clips.append(t_seq[i:i+self.seq_len, :])
                    i += self.seq_len
                clips = [torch.stack(clips[i:i+self.num_seq], 0).transpose(1,2) for i in range(len(clips)+1-self.num_seq)]
                print(clips.__len__(), clips[0].size())
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


class UCF101_3d(data.Dataset):
    def __init__(self,
                 root='/scratch/local/ssd/htd/UCF101',
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq = 1,
                 downsample=3,
                 epsilon=5,
                 num_proposal=3,
                 which_split=1,
                 unit_test=False,
                 load_type='',
                 test_type='c',
                 data_type='rgb'):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.num_proposal = num_proposal
        self.which_split = which_split
        self.unit_test = unit_test
        self.load_type = load_type # define how to sample frames for train/val
        self.test_type = test_type # define how to sample frames for test
        self.data_type = data_type # rgb or flow

        # splits
        if self.data_type == 'rgb':
            if mode == 'train':
                split = '/users/htd/Desktop/SelfSupervision/process_data/data/ucf101/train_split%02d.csv' % self.which_split
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '/users/htd/Desktop/SelfSupervision/process_data/data/ucf101/test_split%02d.csv' % self.which_split # use test for val, temporary
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')
        elif self.data_type == 'flow':
            if mode == 'train':
                split = '/users/htd/Desktop/SelfSupervision/process_data/data/ucf101_flow/train_split%02d.csv' % self.which_split
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '/users/htd/Desktop/SelfSupervision/process_data/data/ucf101_flow/test_split%02d.csv' % self.which_split # use test for val, temporary
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        if not os.path.exists(os.path.join(self.root, 'splits_classification')):
            act_path = '/scratch/shared/nfs1/htd/UCF101' # triton
            if not os.path.exists(os.path.join(act_path, 'splits_classification')):
                act_path = '/scratch/local/ssd/datasets/UCF101' # dev1
        else:
            act_path = self.root

        action_file = os.path.join(act_path, 'splits_classification', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        if not os.path.exists(os.path.join(self.root, 'frame')):
            self.root = '/scratch/shared/nfs1/htd/UCF101'
        if not os.path.exists(os.path.join(self.root, 'frame')):
            self.root = '/scratch/local/ssd/datasets/UCF101'
        print('using data from %s' % self.root)

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if self.unit_test: self.video_info = self.video_info.sample(16, random_state=666)
        # if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)

        # shuffle not required

    def sub_sampler(self, n, vlen, vpath):
        # only works for n=1 now
        result = []
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample) # all possible frames with downsampling
            return [[seq_idx_block, vpath]]
        if self.num_proposal is not None: assert self.num_proposal > n
        n_ = self.num_proposal if self.num_proposal else n
        start_idx_pool = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n_)
        seq_idx_pool = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx_pool

        if self.num_proposal is not None:
            seq_idx_pool = self.sampling_start_idx_with_proposal(seq_idx_pool, vpath, n, p=0.0)

        for idx in range(n):
            seq_idx_block = seq_idx_pool + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
            result.append([seq_idx_block,
                           vpath, # tell which video
                          ])
        return result

    def calc_mean_flow(self, idx_list, vpath):
        mean_flow = []
        fpath = vpath.replace('frame', 'flow', 1)
        for idx in idx_list:
            flow_path = os.path.join(fpath, 'flow_%05d.jpg' % (idx+1))
            img = cv2.imread(flow_path)
            if img is None:
                continue
            mean_flow.append(np.mean(np.abs(img[:,:,0:2] - 127)))

        if mean_flow == []:
            return 0.0
        else:
            return np.mean(mean_flow)

    def sampling_start_idx_with_proposal(self, seq_idx_pool, vpath, n, p=0.0):
        test_idx_pool = seq_idx_pool
        if np.random.rand() < p:
            result = []
            for col_idx in range(test_idx_pool.shape[-1]):
                result.append(self.calc_mean_flow(test_idx_pool[:,col_idx], vpath))
            sort_idx = np.argsort(result)
            idx_pool = [i for i,j in enumerate(sort_idx) if j >= len(sort_idx)-n] # n seq with largest mean flow
        else: # have (1-p) prob to random choice
            idx_pool = np.random.choice([i for i in range(test_idx_pool.shape[-1])], n)
        return seq_idx_pool[:, idx_pool]


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.sub_sampler(1, vlen, vpath)[0] # idx 0 because only select 1 sequence from 1 video
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        # seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        if self.data_type == 'rgb':
            seq = Parallel(n_jobs=1, prefer='threads')\
                   (delayed(pil_loader)(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block) # multiple imgs, a list; jpg if necessary
        elif self.data_type == 'flow':
            seq = Parallel(n_jobs=1, prefer='threads')\
                   (delayed(pil_loader)(os.path.join(vpath, 'flow_%05d.jpg' % (i+1))) for i in idx_block) # multiple imgs, a list; jpg if necessary


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
            if self.test_type == 'a': # obsolete
                # type A: return all available frames
                t_seq = t_seq.view(1, -1, C, H, W).transpose(1,2)
            elif self.test_type == 'b':
                # type B: return all available clips together (variable num_seq)
                SL = t_seq.size(0)
                clips = []; i = 0
                while i+self.seq_len <= SL:
                    clips.append(t_seq[i:i+self.seq_len, :])
                    # i += self.seq_len//2
                    i += self.seq_len
                if num_crop: # if use FineCrop
                    # this part is wrong
                    raise NotImplementedError()
                    NC = len(clips)
                    t_seq = torch.stack(clips, 0).permute(0,2,3,1,4,5).contiguous().view(NC*num_crop, C, self.seq_len, H, W)
                else:
                    t_seq = torch.stack(clips, 0).transpose(1,2)
            elif self.test_type == 'c':
                # type C: return all available clips, but cut into length = num_seq
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


class UCF101_3d_flow(data.Dataset):
    def __init__(self,
                 root='/scratch/local/ssd/htd/UCF101',
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq = 1,
                 downsample=3,
                 epsilon=5,
                 num_proposal=3,
                 which_split=1,
                 unit_test=False,
                 load_type='',
                 test_type='c'):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.num_proposal = num_proposal
        self.which_split = which_split
        self.unit_test = unit_test
        self.load_type = load_type # define how to sample frames for train/val
        self.test_type = test_type # define how to sample frames for test

        # splits
        if mode == 'train':
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/ucf101_flow/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/ucf101_flow/test_split%02d.csv' % self.which_split # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        if not os.path.exists(os.path.join(self.root, 'splits_classification')):
            act_path = '/scratch/shared/nfs1/htd/UCF101' # triton
            if not os.path.exists(os.path.join(act_path, 'splits_classification')):
                act_path = '/scratch/local/ssd/datasets/UCF101' # dev1
        else:
            act_path = self.root

        action_file = os.path.join(act_path, 'splits_classification', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        if not os.path.exists(os.path.join(self.root, 'flow')):
            self.root = '/scratch/shared/nfs1/htd/UCF101'
        if not os.path.exists(os.path.join(self.root, 'flow')):
            self.root = '/scratch/local/ssd/datasets/UCF101'
        print('using data from %s' % self.root)

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if self.unit_test: self.video_info = self.video_info.sample(16, random_state=666)
        # if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)

        # shuffle not required

    def sub_sampler(self, n, vlen, vpath):
        # only works for n=1 now
        result = []
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample) # all possible frames with downsampling
            return [[seq_idx_block, vpath]]
        if self.num_proposal is not None: assert self.num_proposal > n
        n_ = self.num_proposal if self.num_proposal else n
        start_idx_pool = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n_)
        seq_idx_pool = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx_pool

        if self.num_proposal is not None:
            seq_idx_pool = self.sampling_start_idx_with_proposal(seq_idx_pool, vpath, n, p=0.0)

        for idx in range(n):
            seq_idx_block = seq_idx_pool + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
            result.append([seq_idx_block,
                           vpath, # tell which video
                          ])
        return result

    def sampling_start_idx_with_proposal(self, seq_idx_pool, vpath, n, p=0.0):
        test_idx_pool = seq_idx_pool
        if np.random.rand() < p:
            result = []
            for col_idx in range(test_idx_pool.shape[-1]):
                result.append(self.calc_mean_flow(test_idx_pool[:,col_idx], vpath))
            sort_idx = np.argsort(result)
            idx_pool = [i for i,j in enumerate(sort_idx) if j >= len(sort_idx)-n] # n seq with largest mean flow
        else: # have (1-p) prob to random choice
            idx_pool = np.random.choice([i for i in range(test_idx_pool.shape[-1])], n)
        return seq_idx_pool[:, idx_pool]


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.sub_sampler(1, vlen, vpath)[0] # idx 0 because only select 1 sequence from 1 video
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        # seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        seq = Parallel(n_jobs=1, prefer='threads')\
               (delayed(pil_loader)(os.path.join(vpath, 'flow_%05d.jpg' % (i+1))) for i in idx_block) # multiple imgs, a list; jpg if necessary

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
            if self.test_type == 'a': # obsolete
                # type A: return all available frames
                t_seq = t_seq.view(1, -1, C, H, W).transpose(1,2)
            elif self.test_type == 'b':
                # type B: return all available clips together (variable num_seq)
                SL = t_seq.size(0)
                clips = []; i = 0
                while i+self.seq_len <= SL:
                    clips.append(t_seq[i:i+self.seq_len, :])
                    # i += self.seq_len//2
                    i += self.seq_len
                if num_crop: # if use FineCrop
                    # this part is wrong
                    raise NotImplementedError()
                    NC = len(clips)
                    t_seq = torch.stack(clips, 0).permute(0,2,3,1,4,5).contiguous().view(NC*num_crop, C, self.seq_len, H, W)
                else:
                    t_seq = torch.stack(clips, 0).transpose(1,2)
            elif self.test_type == 'c':
                # type C: return all available clips, but cut into length = num_seq
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


class SthsthV1_3d(data.Dataset):
    def __init__(self,
                 root='/scratch/local/ssd/htd/sthsth-v1',
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq = 1,
                 downsample=1, # downsample is not 3, otherwise video too short
                 epsilon=5,
                 num_proposal=3,
                 unit_test=False,
                 test_type='c'):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.num_proposal = num_proposal
        self.unit_test = unit_test
        self.test_type = test_type

        if not os.path.exists(self.root):
            self.root = '/datasets/SomethingSomething'

        print('using data from %s' % self.root)

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join(self.root, 'labels', 'something-something-v1-labels.csv')
        action_df = read_file(action_file)
        for i, act_name in enumerate(action_df):
            self.action_dict_decode[i] = act_name
            self.action_dict_encode[act_name] = i

        # splits
        if mode == 'train':
            split = '~/Desktop/SelfSupervision/process_data/data/sthsth-v1/sthsth-v1-train.csv'
            video_info = pd.read_csv(split, sep=',', header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '~/Desktop/SelfSupervision/process_data/data/sthsth-v1/sthsth-v1-val.csv'
            video_info = pd.read_csv(split, sep=',', header=None)
        else: raise ValueError('wrong mode')

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen, _ = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if self.unit_test: self.video_info = self.video_info.sample(256, random_state=666)
        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)


        # shuffle not required

    def sub_sampler(self, n, vlen, vpath):
        # only works for n=1 now
        result = []
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample) # all possible frames with downsampling
            return [[seq_idx_block, vpath]]
        if self.num_proposal is not None: assert self.num_proposal > n
        n_ = self.num_proposal if self.num_proposal else n
        start_idx_pool = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n_)
        seq_idx_pool = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx_pool

        if self.num_proposal is not None:
            # ddn't use optical flow sampling here
            seq_idx_pool = self.sampling_start_idx_with_proposal(seq_idx_pool, vpath, n, p=0.0)

        for idx in range(n):
            seq_idx_block = seq_idx_pool + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
            result.append([seq_idx_block,
                           vpath, # tell which video
                          ])
        return result

    def calc_mean_flow(self, idx_list, vpath):
        mean_flow = []
        fpath = vpath.replace('frame', 'flow', 1)
        for idx in idx_list:
            flow_path = os.path.join(fpath, 'flow_%05d.jpg' % (idx+1))
            img = cv2.imread(flow_path)
            if img is None:
                continue
            mean_flow.append(np.mean(np.abs(img[:,:,0:2] - 127)))

        if mean_flow == []:
            return 0.0
        else:
            return np.mean(mean_flow)

    def sampling_start_idx_with_proposal(self, seq_idx_pool, vpath, n, p=0.0):
        test_idx_pool = seq_idx_pool
        if np.random.rand() < p:
            result = []
            for col_idx in range(test_idx_pool.shape[-1]):
                result.append(self.calc_mean_flow(test_idx_pool[:,col_idx], vpath))
            sort_idx = np.argsort(result)
            idx_pool = [i for i,j in enumerate(sort_idx) if j >= len(sort_idx)-n] # n seq with largest mean flow
        else: # have (1-p) prob to random choice
            idx_pool = np.random.choice([i for i in range(test_idx_pool.shape[-1])], n)
        return seq_idx_pool[:, idx_pool]


    def __getitem__(self, index):
        vpath, vlen, vid = self.video_info.iloc[index]
        vpath = os.path.join(self.root, vpath)
        items = self.sub_sampler(1, vlen, vpath)[0] # idx 0 because only select 1 sequence from 1 video
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        # seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        seq = Parallel(n_jobs=self.num_seq, prefer='threads')\
               (delayed(pil_loader)(os.path.join(vpath, '%05d.jpg' % (i+1))) for i in idx_block) # multiple imgs, a list; jpg if necessary

        t_seq = self.transform(seq) # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        if self.mode == 'test':
            if self.test_type == 'a':
                # type A: return all available frames
                t_seq = t_seq.view(1, -1, C, H, W).transpose(1,2)
            elif self.test_type == 'b':
                # type B: return all available clips together (variable num_seq)
                SL = t_seq.size(0)
                clips = []; i = 0
                while i+self.seq_len <= SL:
                    clips.append(t_seq[i:i+self.seq_len, :])
                    # i += self.seq_len//2
                    i += self.seq_len
                t_seq = torch.stack(clips, 0).transpose(1,2)
            elif self.test_type == 'c':
                # type C: return all available clips, but cut into length = num_seq
                SL = t_seq.size(0)
                clips = []; i = 0
                while i+self.seq_len <= SL:
                    clips.append(t_seq[i:i+self.seq_len, :])
                    # i += self.seq_len//2
                    i += self.seq_len
                clips = [torch.stack(clips[i:i+self.num_seq], 0).transpose(1,2) for i in range(len(clips)+1-self.num_seq)]
                t_seq = torch.stack(clips, 0)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

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
                 root='/scratch/local/ssd/htd/HMDB51',
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq = 1,
                 downsample=1,
                 epsilon=5,
                 num_proposal=3,
                 which_split=1,
                 unit_test=False,
                 load_type='',
                 test_type='c'):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.num_proposal = num_proposal
        self.which_split = which_split
        self.unit_test = unit_test
        self.load_type = load_type # define how to sample frames for train/val
        self.test_type = test_type # define how to sample frames for test

        # splits
        if mode == 'train':
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/hmdb51/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '/users/htd/Desktop/SelfSupervision/process_data/data/hmdb51/test_split%02d.csv' % self.which_split # use test for val, temporary
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        if not os.path.exists(os.path.join(self.root, 'split')):
            act_path = '/scratch/shared/nfs1/htd/HMDB51' # triton
            if not os.path.exists(os.path.join(act_path, 'split')):
                raise NotImplementedError()
        else:
            act_path = self.root

        action_file = os.path.join(act_path, 'split', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        if not os.path.exists(os.path.join(self.root, 'frame')):
            self.root = '/scratch/local/ssd/datasets/HMDB51'
        if not os.path.exists(os.path.join(self.root, 'frame')):
            raise NotImplementedError()
        print('using data from %s' % self.root)

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if self.unit_test: self.video_info = self.video_info.sample(16, random_state=666)
        # if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)

        # shuffle not required

    def sub_sampler(self, n, vlen, vpath):
        # only works for n=1 now
        result = []
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample) # all possible frames with downsampling
            return [[seq_idx_block, vpath]]
        if self.num_proposal is not None: assert self.num_proposal > n
        n_ = self.num_proposal if self.num_proposal else n
        start_idx_pool = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n_)
        seq_idx_pool = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx_pool

        if self.num_proposal is not None:
            seq_idx_pool = self.sampling_start_idx_with_proposal(seq_idx_pool, vpath, n, p=0.0)

        for idx in range(n):
            seq_idx_block = seq_idx_pool + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
            result.append([seq_idx_block,
                           vpath, # tell which video
                          ])
        return result

    def calc_mean_flow(self, idx_list, vpath):
        mean_flow = []
        fpath = vpath.replace('frame', 'flow', 1)
        for idx in idx_list:
            flow_path = os.path.join(fpath, 'flow_%05d.jpg' % (idx+1))
            img = cv2.imread(flow_path)
            if img is None:
                continue
            mean_flow.append(np.mean(np.abs(img[:,:,0:2] - 127)))

        if mean_flow == []:
            return 0.0
        else:
            return np.mean(mean_flow)

    def sampling_start_idx_with_proposal(self, seq_idx_pool, vpath, n, p=0.0):
        test_idx_pool = seq_idx_pool
        if np.random.rand() < p:
            result = []
            for col_idx in range(test_idx_pool.shape[-1]):
                result.append(self.calc_mean_flow(test_idx_pool[:,col_idx], vpath))
            sort_idx = np.argsort(result)
            idx_pool = [i for i,j in enumerate(sort_idx) if j >= len(sort_idx)-n] # n seq with largest mean flow
        else: # have (1-p) prob to random choice
            idx_pool = np.random.choice([i for i in range(test_idx_pool.shape[-1])], n)
        return seq_idx_pool[:, idx_pool]


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.sub_sampler(1, vlen, vpath)[0] # idx 0 because only select 1 sequence from 1 video
        if items is None: print(vpath)

        idx_block, vpath = items
        if self.mode != 'test':
            assert idx_block.shape == (self.num_seq, self.seq_len)
            idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        # seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        seq = Parallel(n_jobs=self.num_seq, prefer='threads')\
               (delayed(pil_loader)(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block) # multiple imgs, a list; jpg if necessary
        # print(seq[0].size)
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
            if self.test_type == 'a': # obsolete
                # type A: return all available frames
                t_seq = t_seq.view(1, -1, C, H, W).transpose(1,2)
            elif self.test_type == 'b':
                # type B: return all available clips together (variable num_seq)
                SL = t_seq.size(0)
                clips = []; i = 0
                while i+self.seq_len <= SL:
                    clips.append(t_seq[i:i+self.seq_len, :])
                    # i += self.seq_len//2
                    i += self.seq_len
                if num_crop: # if use FiveCrop
                    # this part is wrong
                    raise NotImplementedError()
                    NC = len(clips)
                    t_seq = torch.stack(clips, 0).permute(0,2,3,1,4,5).contiguous().view(NC*num_crop, C, self.seq_len, H, W)
                else:
                    t_seq = torch.stack(clips, 0).transpose(1,2)
            elif self.test_type == 'c':
                # type C: return all available clips, but cut into length = num_seq
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



if __name__ == '__main__':
    transform = transforms.Compose([
        # RandomHorizontalFlip(),
        # RandomRotation(),
        # RandomSizedCrop(size=224),
        RandomCrop(size=224),
        ToTensor(),
        Normalize()
    ])
    datatype = '101'
    if datatype == '101':
        dataset = UCF101_3d(transform = transform, mode='val')
        data_loader = data.DataLoader(dataset,
                                      batch_size=32,
                                      shuffle=False,
                                      num_workers=1)
        print('length of dataset:', len(dataset))
        tic = time.time()
        for i, seq1 in enumerate(data_loader):
            print(i, seq1.size())
            print(i, ':', time.time()-tic, 's')
            tic = time.time()
    elif datatype == 'k':
        print('debugging using kinetics400')
        dataset = Kinetics400_3d(transform = transform,
                                 mode='val')
        my_sampler = data.RandomSampler(dataset)
        data_loader = data.DataLoader(dataset,
                                      batch_size=32,
                                      shuffle=False,
                                      sampler=my_sampler,
                                      num_workers=1)
        print('length of dataset:', len(dataset))
        for i, seq1 in enumerate(data_loader):
            print(i, seq1.size())
            import ipdb; ipdb.set_trace()
