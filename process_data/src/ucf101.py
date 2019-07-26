import os
import csv
import glob

def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            if row: writer.writerow(row)
    print('split saved to %s' % path)

def main():
    '''generate training/testing split, count number of available frames, save in csv'''
    f_root = '/scratch/local/ssd/datasets/UCF101/frame'
    if not os.path.exists(f_root): # triton
        f_root = '/scratch/local/ssd/htd/UCF101/frame'
    data_root = f_root

    split_root = '../data/ucf101/'
    if not os.path.exists(split_root): os.makedirs(split_root)
    train_set = []
    val_set = []

    splits_root = os.path.join('/scratch/local/ssd/datasets/UCF101/splits_classification')
    if not os.path.exists(splits_root): # triton
        splits_root = '/scratch/shared/nfs1/htd/UCF101/splits_classification'

    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        train_split_file = os.path.join(splits_root, 'trainlist%02d.txt' % which_split)
        with open(train_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(data_root, line.split(' ')[0][0:-4]) + '/'
                train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        test_split_file = os.path.join(splits_root, 'testlist%02d.txt' % which_split)
        with open(test_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(data_root, line.rstrip()[0:-4]) + '/'
                test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(split_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(split_root, 'test_split%02d.csv' % which_split))


def main_flow():
    '''generate training/testing split, count number of available frames, save in csv'''
    f_root = '/scratch/local/ssd/datasets/UCF101/flow'
    if not os.path.exists(f_root): # triton
        f_root = '/scratch/local/ssd/htd/UCF101/flow'
    data_root = f_root

    split_root = '../data/ucf101_flow/'
    if not os.path.exists(split_root): os.makedirs(split_root)
    train_set = []
    val_set = []

    splits_root = os.path.join('/scratch/local/ssd/datasets/UCF101/splits_classification')
    if not os.path.exists(splits_root): # triton
        splits_root = '/scratch/shared/nfs1/htd/UCF101/splits_classification'

    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        train_split_file = os.path.join(splits_root, 'trainlist%02d.txt' % which_split)
        with open(train_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(data_root, line.split(' ')[0][0:-4]) + '/'
                train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        test_split_file = os.path.join(splits_root, 'testlist%02d.txt' % which_split)
        with open(test_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(data_root, line.rstrip()[0:-4]) + '/'
                test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(split_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(split_root, 'test_split%02d.csv' % which_split))

if __name__ == '__main__':
    # main()
    main_flow()
