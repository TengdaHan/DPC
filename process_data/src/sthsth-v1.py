import os 
import csv 
import glob 
import pandas as pd 
from tqdm import tqdm
from joblib import Parallel, delayed

def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            if row: writer.writerow(row)
    print('split saved to %s' % path)

def read_csv(path, delimiter=','):
    with open(path, 'r') as f:
        writer = csv.writer(f, delimiter=delimiter)
        content = []
        for row in writer:
            content.append(row)
    return content 

def read_file(path,):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content 

def work(df, ):
    data = []
    for i, row in df.iterrows():
        [video_id, act_name] = row
        v_path = os.path.join(root, 'Frames', str(video_id))
        num_frame = len(glob.glob(os.path.join(v_path, '*.jpg')))
        assert num_frame != 0 
        short_v_path = os.path.join('Frames', str(video_id))
        data.append([short_v_path, num_frame, action_dict_encode[act_name]])
    return data

def process(path):
    print('processing file: %s' % path)
    df = pd.read_csv(path, sep=';', header=None)
    n = 1000
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]
    processed_list = Parallel(n_jobs=16)(delayed(work)(i) for i in tqdm(list_df,total=len(list_df)))
    result = [item for sublist in processed_list for item in sublist]
    assert len(result) == len(df)
    return result 

def main():
    global root
    root = '/datasets/SomethingSomething'

    # get action list
    global action_dict_encode, action_dict_decode
    action_dict_encode = {}
    action_dict_decode = {}
    action_file = os.path.join(root, 'labels', 'something-something-v1-labels.csv')
    action_df = read_file(action_file)
    for i, act_name in enumerate(action_df):
        action_dict_decode[i] = act_name
        action_dict_encode[act_name] = i

    split_root = '../data/sthsth-v1/'
    if not os.path.exists(split_root): os.makedirs(split_root)

    train_file = os.path.join(root, 'labels', 'something-something-v1-train.csv')
    train_data = process(train_file)
    write_list(train_data, os.path.join(split_root, 'sthsth-v1-train.csv'))

    val_file = os.path.join(root, 'labels', 'something-something-v1-validation.csv')
    val_data = process(val_file)
    write_list(val_data, os.path.join(split_root, 'sthsth-v1-val.csv'))

if __name__ == '__main__':
    main()
