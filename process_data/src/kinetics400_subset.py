import os
import pandas as pd
import csv
from tqdm import tqdm
# from joblib import Parallel, delayed
import time
import glob

# For tiny kinetics dataset
def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            if row:
                writer.writerow(row)
    print('split saved to %s' % path)

def main():
    data_root = '/scratch/local/ssd/htd/kinetics400/frame_subset/'
    split_root = '../data/kinetics400_subset/'
    if not os.path.exists(split_root): os.makedirs(split_root)
    train_set = []
    val_set = []

    v_class_paths = sorted(glob.glob(os.path.join(data_root, '*/')))
    for v_class in tqdm(v_class_paths, total=len(v_class_paths)):
        v_paths = sorted(glob.glob(os.path.join(v_class, '*/')))
        if len(v_paths) == 2:
            train_set.append([v_paths[0], len(glob.glob(os.path.join(v_paths[0], '*.jpg')))])
            val_set.append([v_paths[1], len(glob.glob(os.path.join(v_paths[1], '*.jpg')))])
        else:
            assert len(v_paths) > 100 
            train_set.extend([[v_paths[i], len(glob.glob(os.path.join(v_paths[i], '*.jpg')))] for i in range(0, 100)])
            val_set.extend([[v_paths[i], len(glob.glob(os.path.join(v_paths[i], '*.jpg')))] for i in range(100, len(v_paths))])

    # val_set = val_set[0:200] # cut half, enough

    write_list(train_set, os.path.join(split_root, 'train_split.csv'))
    write_list(val_set, os.path.join(split_root, 'val_split.csv'))

def main_separate():
    data_root = '/scratch/local/ssd/htd/kinetics400/frame_subset/'
    split_root = '../data/kinetics400_subset/'
    if not os.path.exists(split_root): os.makedirs(split_root)
    train_set = []
    val_set = []

    for mode in ['train_split', 'val_split']:
        print('working on %s mode' % mode)
        v_class_paths = sorted(glob.glob(os.path.join(data_root, mode, '*/')))
        for v_class in tqdm(v_class_paths, total=len(v_class_paths)):
            v_paths = sorted(glob.glob(os.path.join(v_class, '*/')))

            if mode == 'train_split':
                train_set.extend([[p, len(glob.glob(os.path.join(p, '*.jpg')))] for p in v_paths])
            else:
                val_set.extend([[p, len(glob.glob(os.path.join(p, '*.jpg')))] for p in v_paths])

    # val_set = val_set[0:200] # cut half, enough

    write_list(train_set, os.path.join(split_root, 'train_split.csv'))
    write_list(val_set, os.path.join(split_root, 'val_split.csv'))

if __name__ == '__main__':
    main_separate()
    # save_split('val')
    # test doesn't work because no label
