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
    f_root = '/scratch/local/ssd/htd/HMDB51/frame'
    if not os.path.exists(f_root): # triton
        raise NotImplementedError()
        f_root = '/scratch/local/ssd/htd/UCF101/frame'
    data_root = f_root

    split_root = '../data/hmdb51/'
    if not os.path.exists(split_root): os.makedirs(split_root)
    train_set = []
    val_set = []

    splits_root = os.path.join('/scratch/shared/nfs1/htd/HMDB51/split/testTrainMulti_7030_splits')
    if not os.path.exists(splits_root): # triton
        raise NotImplementedError()

    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        split_files = sorted(glob.glob(os.path.join(splits_root, '*_test_split%d.txt' % which_split)))
        assert len(split_files) == 51
        for split_file in split_files:
            action_name = os.path.basename(split_file)[0:-16]
            with open(split_file, 'r') as f:
                for line in f:
                    video_name = line.split(' ')[0]
                    _type = line.split(' ')[1]
                    vpath = os.path.join(data_root, action_name, video_name[0:-4]) + '/'
                    if _type == '1':
                        train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])
                    elif _type == '2':
                        test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(split_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(split_root, 'test_split%02d.csv' % which_split))

if __name__ == '__main__':
    main()
