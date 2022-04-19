import os
import torch
import time
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy
from functools import partial
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import SiameseNetwork_ete
from loss import ContrastiveLoss
from sklearn.metrics import confusion_matrix, classification_report
from preprocessing import get_preproc, get_segmentation
from metrics import AvgLoss, Accuracy, Sensitivity, Specificity, print_metrics, reset_metrics
from detector_utils import load, modified_dtw, approx_exp_moving_avg, compute_binary_accuracy,load_for_original

max_epochs = 100
batch_size = 256
n_intervals = 10
dataset_path = '/home/swyoo/diff_cnn/mitbih_database/'

# checkpoint
checkpoint_name = 'test.pth'
device = torch.device('cuda:7')

patients = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

k = 20
n_segments = k
std_dist = 0.1 # std 구하기
std_inter = 100 # std 구하기
tp = 0
tn = 0
fp = 0
fn = 0

for idx, patient in enumerate(patients):
    print(patient)
    sig, annotations = load_for_original(dataset_path, patient, from_npy=False) # 각 점마다의 time, type, sample#
    #print(sig[0])
    #print(annotations[0])
    preproc = partial(get_preproc('bandpass'), fc=(1.5, 150))
    sig_f = preproc(sig) # bandpass
    segmentation = get_segmentation('windowing')
    segments, annotations = segmentation(sig_f, annotations)
    #print(len(segments))
    #print(len(annotations))
    X = []
    for segment in segments:
        segment = (segment - segment.min())/(segment.max()-segment.min())
        X.append(segment)
    assert len(X) == len(annotations) # 100번 환자 첫번째부터 안 들어가는 이유는 앞 부분이 너무 짧아
    #print(annotations[0]['Sample#'])
    
    trues = []
    preds = []
    
    for i in range(0,len(X)):
        #print(annotations[i])
        trues.append(annotations[i]['type'])
        preds.append(annotations[i]['pred'])
    #print(len(trues))
    #print(len(preds))
    normal_num = 0  
    for pred, truth in zip(preds, trues):
        #print(pred)
        #print(truth)

        if truth == 0:
            normal_num += 1
            if pred == 'N':
                # true negative
                tn += 1 
            elif pred == 'A':
                # false positive
                fp += 1
        else:
            assert truth is not None
            if pred == 'N':
                # false negative
                fn += 1
            elif pred == 'A':
                # true positive
                tp += 1
    #print(normal_num)
print(tn, tp, fn, fp)
se, sp, acc = compute_binary_accuracy(tn, tp, fn, fp)
print(se, sp, acc)
    
