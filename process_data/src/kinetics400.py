import os
import pandas as pd
import csv
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import glob

## multiple source version: 
# data_root = ['/scratch/local/ssd/htd/kinetics400',
#              '/scratch/local/ramdisk/htd/kinetics400'] # must be list

# def get_split(data_root, split_path, mode):
#     print('processing %s split ...' % mode)
#     split_content = pd.read_csv(split_path).iloc[:,0:4]
#     split_master_list = []
#     for dr in data_root:
#         split_list = Parallel(n_jobs=64)\
#                      (delayed(check_exists)(row, dr) \
#                      for i, row in tqdm(split_content.iterrows(), total=len(split_content)))
#         split_master_list.extend(split_list)
#     return split_master_list

# data_root = '/datasets/kinetics_jpg'
data_root = '/scratch/local/ssd/htd/kinetics400/frame_full'

def get_split(root, split_path, mode):
    print('processing %s split ...' % mode)
    print('checking %s' % root)
    split_list = []
    split_content = pd.read_csv(split_path).iloc[:,0:4]
    split_list = Parallel(n_jobs=64)\
                 (delayed(check_exists)(row, root) \
                 for i, row in tqdm(split_content.iterrows(), total=len(split_content)))
    return split_list

def check_exists(row, root):
    dirname = '_'.join([row['youtube_id'], '%06d' % row['time_start'], '%06d' % row['time_end']])
    full_dirname = os.path.join(root, row['label'], dirname)
    if os.path.exists(full_dirname):
        n_frames = len(glob.glob(os.path.join(full_dirname, '*.jpg')))
        return [full_dirname, n_frames]
    else:
        return None

def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            if row:
                writer.writerow(row)
    print('split saved to %s' % path)

def save_split(mode):
    train_split_path = '/users/htd/Data/Kinetics/kinetics_train/kinetics_train.csv'
    val_split_path = '/users/htd/Data/Kinetics/kinetics_val/kinetics_val.csv'
    test_split_path = '/users/htd/Data/Kinetics/kinetics_test/kinetics_test.csv'

    output_path = '/users/htd/Desktop/SelfSupervision/process_data/data/kinetics400'
    if not os.path.exists(output_path): os.makedirs(output_path)
    if mode == 'train':
        train_split = get_split(os.path.join(data_root, 'train_split'), train_split_path, 'train')
        write_list(train_split, os.path.join(output_path, 'train_split.csv'))
    elif mode == 'val':
        val_split = get_split(os.path.join(data_root, 'val_split'), val_split_path, 'val')
        write_list(val_split, os.path.join(output_path, 'val_split.csv'))
    elif mode == 'test':
        test_split = get_split(data_root, test_split_path, 'test')
        write_list(test_split, os.path.join(output_path, 'test_split.csv'))
    else:
        raise IOError('wrong mode')


if __name__ == '__main__':
    save_split('val')
    save_split('train')
    
    # test doesn't work because no label
