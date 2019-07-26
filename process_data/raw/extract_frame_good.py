# by htd@robots.ox.ac.uk

from joblib import delayed, Parallel
import os 
import sys 
import glob 
import subprocess
from tqdm import tqdm 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def extract_video_ffmpeg(v_path, f_root):
    '''ffmpeg version: --- not very good on parallel
       v_path: single video path;
       f_root: root to store frames'''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(f_root, v_class, v_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cmd = ['ffmpeg', '-i', '%s' % v_path,
           '-f', 'image2',
           '-s', '320x240',
           '%s' % os.path.join(out_dir, 'image_%05d.png')]
    ffmpeg = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        out, err = ffmpeg.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        ffmpeg.kill()
        try:
            out, err = ffmpeg.communicate()
        except ValueError: # otherwise, ValueError: Invalid file object ...
            out = b''
            err = b''
    cmd1 = ['ffprobe', '-v', 'error',
           '-select_streams', 'v:0',
           '-show_entries', 'stream=nb_frames',
           '-of', 'default=nokey=1:noprint_wrappers=1',
           v_path]
    ffprobe = subprocess.Popen(cmd1, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffprobe.communicate()
    nb_frames = out.decode().splitlines()[0]
    # print('%s extracted' % v_path)
    if not os.path.exists(os.path.join(out_dir, 'image_%05d.png' % (int(nb_frames)-1))):
        # import ipdb; ipdb.set_trace()
        print(out_dir, 'is not extracted successfully')
    # deal with end of video - broken frame - issue
    if os.path.exists(os.path.join(out_dir, 'image_%05d.png' % int(nb_frames))):
        os.remove(os.path.join(out_dir, 'image_%05d.png' % int(nb_frames)))

def extract_video_opencv(v_path, f_root, dim=240):
    '''opencv version: relatively good
       v_path: single video path;
       f_root: root to store frames'''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(f_root, v_class, v_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height==0): 
        print(v_path, 'not successfully loaded, drop ..'); return
    new_dim = resize_dim(width, height, dim)

    success, image = vidcap.read()
    count = 1
    while success:
        image = cv2.resize(image, new_dim, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out_dir, 'image_%05d.jpg' % count), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])# quality from 0-100, 95 is default, high is good
        success, image = vidcap.read()
        count += 1
    if nb_frames > count:
        print('/'.join(out_dir.split('/')[-2::]), 'NOT extracted successfully: %df/%df' % (count, nb_frames))
    vidcap.release()

def extract_ff_opencv(v_path, frame_root, flow_root, dim=240):
    '''opencv version: 
       v_path: single video path;
       f_root: root to store flow'''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    frame_out_dir = os.path.join(frame_root, v_class, v_name)
    flow_out_dir = os.path.join(flow_root, v_class, v_name)
    for i in [frame_out_dir, flow_out_dir]:
        if not os.path.exists(i): os.makedirs(i)

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height==0): 
        print(width, height, v_path); import ipdb; ipdb.set_trace()
    new_dim = resize_dim(width, height, dim)

    success, image = vidcap.read()
    count = 1
    flows = np.zeros((nb_frames, *new_dim[::-1], 2)) # save flow as one file, t=0 is all 0 # N,H,W,C
    while success:
        image = cv2.resize(image, new_dim, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(frame_out_dir, 'image_%05d.jpg' % count), 
                    image, 
                    [cv2.IMWRITE_JPEG_QUALITY, 90]) # quality from 0-100, 95 is default, high is good
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if count != 1:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, image_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows[count-1,:] = flow
        prev_gray = image_gray 
        success, image = vidcap.read()
        count += 1
    if nb_frames > count:
        print(frame_out_dir, 'is NOT extracted successfully')
    else:
        flows = (flows / max(flows.max(), abs(flows.min())) * 127.5 + 127.5).astype(np.uint8) # np.int8 is wrong, must use uint8
        zero_channel = np.zeros((*new_dim[::-1],1)).astype(np.uint8)
        for idx in range(nb_frames):
            cv2.imwrite(os.path.join(flow_out_dir, 'flow_%05d.jpg' % (idx+1)),
                        np.concatenate((flows[idx,:], zero_channel), -1),
                        [cv2.IMWRITE_JPEG_QUALITY, 50]) # quality from 0-100, 95 is default, high is good
    vidcap.release()

def extract_flow_opencv(v_path, flow_root, dim=240):
    '''opencv version: 
       v_path: single video path;
       f_root: root to store flow'''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    flow_out_dir = os.path.join(flow_root, v_class, v_name)
    if not os.path.exists(flow_out_dir): os.makedirs(flow_out_dir)

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height==0): 
        print(width, height, v_path); import ipdb; ipdb.set_trace()
    new_dim = resize_dim(width, height, dim)

    success, image = vidcap.read()
    count = 1
    flows = np.zeros((nb_frames, *new_dim[::-1], 2)) # save flow as one file, t=0 is all 0 # N,H,W,C
    while success:
        image = cv2.resize(image, new_dim, interpolation = cv2.INTER_LINEAR)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if count != 1:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, image_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows[count-1,:] = flow
        prev_gray = image_gray 
        success, image = vidcap.read()
        count += 1
    if nb_frames > count:
        print(flow_out_dir, 'NOT extracted successfully: %df / %df' % (count, nb_frames))
    
    flows = (flows / max(flows.max(), abs(flows.min())) * 127.5 + 127.5).astype(np.uint8) # np.int8 is wrong, must use uint8
    zero_channel = np.zeros((*new_dim[::-1],1)).astype(np.uint8)
    for idx in range(nb_frames):
        if os.path.exists(os.path.join(flow_out_dir, 'flow_%05d.jpg' % (idx+1))): continue
        cv2.imwrite(os.path.join(flow_out_dir, 'flow_%05d.jpg' % (idx+1)),
                    np.concatenate((flows[idx,:], zero_channel), -1),
                    [cv2.IMWRITE_JPEG_QUALITY, 90]) # quality from 0-100, 95 is default, high is good
    vidcap.release()

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

def main_UCF101(mode='frame'):
    print('extracting %s ... ' % mode)
    if mode == 'frame':
        v_root = '/scratch/local/ssd/datasets/UCF101/videos'
        f_root = '/scratch/local/ssd/datasets/UCF101/frame'
        if not os.path.exists(v_root): # triton
            v_root = '/scratch/shared/nfs1/htd/UCF101/videos'
            f_root = '/scratch/local/ssd/htd/UCF101/frame'
        if not os.path.exists(v_root): # use given videos
            v_root = '/datasets/UCF101/videos'
            f_root = '/scratch/local/ssd/htd/UCF101/frame'

        print('using dataset from %s' % v_root)
        print('frame save to %s' % f_root)
        
        if not os.path.exists(f_root): os.makedirs(f_root)

        v_act_root = glob.glob(os.path.join(v_root, '*/'))
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.avi'))
            v_paths = sorted(v_paths)
            Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root) for p in v_paths)
    elif mode == 'flow':
        v_root = '/scratch/local/ssd/datasets/UCF101/videos'
        flow_root = '/scratch/local/ssd/htd/UCF101/flow'
        if not os.path.exists(v_root): # triton
            v_root = '/scratch/shared/nfs1/htd/UCF101/videos'
        if not os.path.exists(v_root):
            v_root = '/datasets/UCF101/videos'

        print('using dataset from %s' % v_root)
        print('flow save to %s' % flow_root)
        
        if not os.path.exists(flow_root): os.makedirs(flow_root)
        v_act_root = glob.glob(os.path.join(v_root, '*/'))
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.avi'))
            v_paths = sorted(v_paths)
            Parallel(n_jobs=32)(delayed(extract_flow_opencv)(p, flow_root) for p in tqdm(v_paths,total=len(v_paths)))
    elif mode == 'ff':
        v_root = '/scratch/local/ssd/datasets/UCF101/videos'
        frame_root = '/scratch/local/ssd/datasets/UCF101/frame'
        flow_root = '/scratch/local/ssd/datasets/UCF101/flow'
        if not os.path.exists(v_root): # triton
            v_root = '/scratch/shared/nfs1/htd/UCF101/videos'
            frame_root = '/scratch/local/ssd/htd/UCF101/frame'
            flow_root = '/scratch/local/ssd/htd/UCF101/flow'
        if not os.path.exists(v_root):
            v_root = '/datasets/UCF101/videos'
            frame_root = '/scratch/local/ssd/htd/UCF101/frame'
            flow_root = '/scratch/local/ssd/htd/UCF101/flow'

        print('using dataset from %s' % v_root)
        print('frame save to %s' % frame_root)
        print('flow save to %s' % flow_root)
        
        if not os.path.exists(frame_root): os.makedirs(frame_root)
        if not os.path.exists(flow_root): os.makedirs(flow_root)
        v_act_root = glob.glob(os.path.join(v_root, '*/'))
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.avi'))
            v_paths = sorted(v_paths)
            Parallel(n_jobs=32)(delayed(extract_ff_opencv)(p, frame_root, flow_root) for p in v_paths)

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
