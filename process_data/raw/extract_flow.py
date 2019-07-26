# by htd@robots.ox.ac.uk

from joblib import delayed, Parallel
import os 
import sys 
import glob 
import subprocess
from tqdm import tqdm 
from PIL import Image
import cv2
import numpy as np 
import sys
sys.path.append('/users/htd/Documents/pyflow/')
import pyflow 
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time  


# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


def extract_flow(v_path, flow_root, dim=100):
    '''pyflow version: 
       v_path: single frame path;
       f_root: root to store flow'''
    
    v_class = v_path.split('/')[-3]
    v_name = v_path.split('/')[-2]
    flow_out_dir = os.path.join(flow_root, v_class, v_name)
    if os.path.exists(flow_out_dir): print('detach from:', flow_out_dir); return
    else: os.makedirs(flow_out_dir)
    frame_list = sorted(glob.glob(os.path.join(v_path, '*.jpg')))
    nb_frames = len(frame_list)
    flows = None; new_dim = None 

    for i in range(nb_frames-1):
        im1 = cv2.imread(frame_list[i])
        im2 = cv2.imread(frame_list[i+1])
        if i == 0:
            height, width = im1.shape[0:2]
            new_dim = resize_dim(width, height, dim)
            flows = np.zeros((nb_frames, *new_dim[::-1], 2))
        im1 = cv2.resize(im1, new_dim)
        im2 = cv2.resize(im2, new_dim)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        im1 = im1.astype(float) / 255. 
        im2 = im2.astype(float) / 255. 
        u, v, _ = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        flows[i+1] = flow 

    boundary = max(np.abs(flows.max()), np.abs(flows.min()))
    scale = 127.5/boundary if boundary != 0.0 else 1.0
    zero_channel = np.zeros((nb_frames,*new_dim[::-1],1)).astype(np.uint8)
    flows = np.concatenate((flows, zero_channel), -1)
    
    flows = (flows * scale + 127.5).astype(np.uint8) # np.int8 is wrong, must use uint8
    
    for idx in range(nb_frames):
        if os.path.exists(os.path.join(flow_out_dir, 'flow_%05d.jpg' % (idx+1))): continue
        cv2.imwrite(os.path.join(flow_out_dir, 'flow_%05d.jpg' % (idx+1)),
                    flows[idx,:], [cv2.IMWRITE_JPEG_QUALITY, 90]) # quality from 0-100, 95 is default, high is good
    
    scale_file = open(os.path.join(flow_out_dir, 'scale.txt'), 'w')
    scale_file.write(str(scale))
    scale_file.close()


def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w)) 

## Visualization ##
def plot_flow(flow, filename):
    c = flow.shape[-1]
    for i in range(c):
        plt.subplot(2, 2, i+1)
        plt.imshow(flow[:,:,i])
        plt.title('channel %d' %(i+1))
        plt.subplot(2, 2, i+3)
        plt.hist(flow[:,:,i].ravel(), 256)
        plt.title('hist of channel %d' %(i+1))

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def plot_flow_with_raw(prev_gray, next_gray, flow, filename):
    plt.subplot(2, 2, 1)
    plt.imshow(flow[:,:,0])
    plt.title('channel 1')
    plt.subplot(2, 2, 2)
    plt.imshow(flow[:,:,1])
    plt.title('channel 2')

    plt.subplot(2, 2, 3)
    plt.imshow(prev_gray)
    plt.title('prev')
    plt.subplot(2, 2, 4)
    plt.imshow(next_gray)
    plt.title('next')

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

###################

def main_UCF101(mode='flow'):
    print('extracting %s ... ' % mode)
    if mode == 'flow':
        v_root = '/scratch/local/ssd/htd/UCF101/frame'
        flow_root = '/scratch/local/ssd/htd/UCF101/tmp'
        if not os.path.exists(v_root): # triton
            import ipdb; ipdb.set_trace()

        print('using frame from %s' % v_root)
        print('flow save to %s' % flow_root)
        if not os.path.exists(flow_root): os.makedirs(flow_root)

        v_act_root = sorted(glob.glob(os.path.join(v_root, '*/')))
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*/'))
            v_paths = sorted(v_paths)
            Parallel(n_jobs=32)(delayed(extract_flow)(p, flow_root) for p in tqdm(v_paths,total=len(v_paths)))

# not completed: 
def main_kinetics(mode='frame'):
    print('extracting %s ... ' % mode)
    for basename in ['train_split', 'val_split']:
        if mode == 'frame':
            v_root = '/scratch/local/ssd/htd/kinetics400/video_subset' + '/' + basename
            if not os.path.exists(v_root):
                v_root = '/scratch/shared/nfs1/htd/kinetics400/video_subset' + '/' + basename
            f_root = '/scratch/local/ssd/htd/kinetics400/frame_subset' + '/' + basename 

            if not os.path.exists(f_root): os.makedirs(f_root)
            v_act_root = glob.glob(os.path.join(v_root, '*/'))

            for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
                v_paths = glob.glob(os.path.join(j, '*.mp4'))
                v_paths = sorted(v_paths)
                Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root, dim=128) for p in v_paths)
                # [extract_video_opencv(p, f_root) for p in v_paths]
        elif mode == 'ff':
            v_root = '/scratch/local/ssd/htd/kinetics400/video_subset' + '/' + basename
            if not os.path.exists(v_root):
                v_root = '/scratch/shared/nfs1/htd/kinetics400/video_subset' + '/' + basename
            frame_root = '/scratch/local/ssd/htd/kinetics400/frame_subset' + '/' + basename 
            flow_root = '/scratch/local/ssd/htd/kinetics400/flow_subset' + '/' + basename 

            if not os.path.exists(frame_root): os.makedirs(frame_root)
            if not os.path.exists(flow_root): os.makedirs(flow_root)
            v_act_root = glob.glob(os.path.join(v_root, '*/'))

            for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
                v_paths = glob.glob(os.path.join(j, '*.mp4'))
                v_paths = sorted(v_paths)
                Parallel(n_jobs=32)(delayed(extract_ff_opencv)(p, frame_root, flow_root, dim=128) for p in v_paths)
                # [extract_video_opencv(p, f_root) for p in v_paths]
        else:
            raise ValueError('invalid mode')

def main_kinetics_full(mode='frame'):
    print('extracting %s ... ' % mode)
    for basename in ['train_split', 'val_split']:
        if basename == 'val_split': print('val split already done'); continue # val_split already done
        if mode == 'frame':
            v_root = '/datasets/KineticsClean' + '/' + basename
            if not os.path.exists(v_root):
                print('Wrong v_root')
                import ipdb; ipdb.set_trace()
            f_root = '/scratch/local/ssd/htd/kinetics400/frame_full' + '/' + basename 
            # f_root = '/scratch/local/ssd/htd/kinetics400/frame_full' + '/' + basename 
            print('Extract to: \nframe: %s' % f_root)
            if not os.path.exists(f_root): os.makedirs(f_root)
            v_act_root = glob.glob(os.path.join(v_root, '*/'))
            v_act_root = sorted(v_act_root)

            # if resume, remember to delete the last video folder
            for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
                v_paths = glob.glob(os.path.join(j, '*.mp4'))
                v_paths = sorted(v_paths)
                # for resume:
                v_class = j.split('/')[-2]
                out_dir = os.path.join(f_root, v_class)
                if os.path.exists(out_dir): print(out_dir, 'exists!'); continue
                print('extracting: %s' % v_class)
                # n_jobs=32 is a good choice, if too big, vidcap will fail to read video, idk why
                Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root, dim=150) for p in tqdm(v_paths, total=len(v_paths))) # tried 150 and 256
                # [extract_video_opencv(p, f_root) for p in v_paths]
        elif mode == 'ff':
            v_root = '/datasets/KineticsClean' + '/' + basename
            if not os.path.exists(v_root):
                print('Wrong v_root')
                import ipdb; ipdb.set_trace()
            frame_root = '/scratch/shared/nfs1/htd/kinetics400/frame_full' + '/' + basename 
            flow_root = '/scratch/shared/nfs1/htd/kinetics400/flow_full' + '/' + basename 
            print('Extract to: \nframe: %s\nflow: %s' % (frame_root, flow_root))
            if not os.path.exists(frame_root): os.makedirs(frame_root)
            if not os.path.exists(flow_root): os.makedirs(flow_root)
            v_act_root = glob.glob(os.path.join(v_root, '*/'))
            v_act_root = sorted(v_act_root)

            # if resume, remember to delete the last video&flow folder
            for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
                v_paths = glob.glob(os.path.join(j, '*.mp4'))
                v_paths = sorted(v_paths)
                # for resume:
                v_class = j.split('/')[-2]
                out_dir = os.path.join(frame_root, v_class)
                if os.path.exists(out_dir): print(out_dir, 'exists!'); continue
                print('extracting: %s' % v_class)
                Parallel(n_jobs=32)(delayed(extract_ff_opencv)(p, frame_root, flow_root, dim=150) for p in tqdm(v_paths, total=len(v_paths)))
                # [extract_video_opencv(p, f_root) for p in v_paths]
        else:
            raise ValueError('invalid mode')

def main_HMDB51(mode='frame'):
    print('extracting %s ... ' % mode)
    if mode == 'frame':
        v_root = '/scratch/local/ssd/htd/HMDB51/videos'
        f_root = '/scratch/local/ssd/htd/HMDB51/frame'
        if not os.path.exists(v_root): # triton
            v_root = '/scratch/shared/nfs1/htd/HMDB51/videos'
        if not os.path.exists(v_root): # use given videos
            raise NotImplementedError()
            v_root = '/datasets/UCF101/videos'

        print('using dataset from %s' % v_root)
        print('frame save to %s' % f_root)
        
        if not os.path.exists(f_root): os.makedirs(f_root)

        v_act_root = glob.glob(os.path.join(v_root, '*/'))
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.avi'))
            v_paths = sorted(v_paths)
            Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root) for p in tqdm(v_paths,total=len(v_paths)))
    elif mode == 'flow':
        v_root = '/scratch/local/ssd/datasets/HMDB51/videos'
        flow_root = '/scratch/local/ssd/htd/HMDB51/flow'
        if not os.path.exists(v_root): # triton
            v_root = '/scratch/shared/nfs1/htd/HMDB51/videos'
        if not os.path.exists(v_root):
            raise NotImplementedError()
            v_root = '/datasets/HMDB51/videos'

        print('using dataset from %s' % v_root)
        print('flow save to %s' % flow_root)
        
        if not os.path.exists(flow_root): os.makedirs(flow_root)
        v_act_root = glob.glob(os.path.join(v_root, '*/'))
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.avi'))
            v_paths = sorted(v_paths)
            Parallel(n_jobs=32)(delayed(extract_flow_opencv)(p, flow_root) for p in tqdm(v_paths,total=len(v_paths)))
    elif mode == 'ff':
        v_root = '/scratch/local/ssd/datasets/HMDB51/videos'
        frame_root = '/scratch/local/ssd/htd/HMDB51/frame'
        flow_root = '/scratch/local/ssd/htd/HMDB51/flow'
        if not os.path.exists(v_root): # triton
            v_root = '/scratch/shared/nfs1/htd/HMDB51/videos'
        if not os.path.exists(v_root):
            raise NotImplementedError()
            v_root = '/datasets/HMDB51/videos'

        print('using dataset from %s' % v_root)
        print('frame save to %s' % frame_root)
        print('flow save to %s' % flow_root)
        
        if not os.path.exists(frame_root): os.makedirs(frame_root)
        if not os.path.exists(flow_root): os.makedirs(flow_root)
        v_act_root = glob.glob(os.path.join(v_root, '*/'))
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.avi'))
            v_paths = sorted(v_paths)
            Parallel(n_jobs=32)(delayed(extract_ff_opencv)(p, frame_root, flow_root) for p in tqdm(v_paths,total=len(v_paths)))


if __name__ == '__main__':
    # main_kinetics(mode='ff')
    main_UCF101(mode='flow')
    # main_HMDB51(mode='frame')
    # main_kinetics_full(mode='frame')
