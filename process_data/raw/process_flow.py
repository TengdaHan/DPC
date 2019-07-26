from joblib import delayed, Parallel
import os 
import sys 
import glob 
from tqdm import tqdm 
import cv2
import numpy as np 

def summarize_flow(flow_root, out_root):
    '''summarize video optical flow into a mean vector'''
    for basename in ['train_split', 'val_split']:
        flow_root_ = flow_root + '/' + basename
        pass 


if __name__ == '__main__':
    summarize_flow(flow_root='/scratch/local/ssd/htd/kinetics400/flow_subset',
                   out_root='')