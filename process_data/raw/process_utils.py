import subprocess 
import os 
import glob
from tqdm import tqdm 
# from joblib import delayed, Parallel
# no parallel because small dataset

def copy_paste(file_path, target_path):
    file_name = os.path.basename(file_path)
    cmd = ['cp', file_path, target_path]
    copy = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        out, err = copy.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        copy.kill()
        out, err = copy.communicate()
    # check
    if not os.path.exists(os.path.join(target_path, file_name)):
        print(file_path, 'Not copy-paste successfully')

def copy_paste_all(size=None):
    data_root = '/datasets/Kinetics'
    t_root = '../video_subset'
    class_path = glob.glob(os.path.join(data_root, '*/'))
    assert len(class_path) == 400
    if size:
        class_path = sorted(class_path)[0:size]
    for p in tqdm(class_path, total=len(class_path)):
        try:
            if size:
                mp4_list = sorted(glob.glob(os.path.join(p, '*.mp4')))
                if len(mp4_list) < size:
                    import ipdb; ipdb.set_trace()
                else:
                    v_path = sorted(glob.glob(os.path.join(p, '*.mp4')))[0:round(size*1.5)] # pick the first N+0.5N (as val set)
            else:
                v_path = sorted(glob.glob(os.path.join(p, '*.mp4')))[0:2] # pick the first two
        except:
            import ipdb; ipdb.set_trace()
        for i in v_path: 
            [class_name, video_name] = i.split('/')[-2::]
            t_dir = os.path.join(t_root, class_name)
            if not os.path.exists(t_dir): os.makedirs(t_dir)
            t_path = os.path.join(t_dir, video_name)
            if not os.path.exists(t_path): copy_paste(i, t_dir)

def extract_video(v_path, f_root):
    '''v_path: single video path;
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
        out, err = ffmpeg.communicate()
    # print('%s extracted' % v_path)
    if not os.path.exists(os.path.join(out_dir, 'image_00001.png')):
        print(out_dir, 'is not extracted successfully')

def extract_all():
    v_root = '../video_subset'
    f_root = '/scratch/local/ssd/htd/kinetics400/frames_subset'
    if not os.path.exists(f_root): os.makedirs(f_root)
    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    assert len(v_act_root) == 400

    for j in tqdm(v_act_root, total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.mp4'))
        v_paths = sorted(v_paths)
        # Parallel(n_jobs=32)(delayed(extract_video)(p, f_root) for p in v_paths)
        [extract_video(p, f_root) for p in v_paths]


if __name__ == '__main__':
    copy_paste_all(size=40)
    # extract_all()
