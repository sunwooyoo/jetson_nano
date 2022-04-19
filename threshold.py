import os
import torch
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
from detector_utils import load, modified_dtw, approx_exp_moving_avg

max_epochs = 100
batch_size = 256
n_intervals = 10
dataset_path = '/home/swyoo/diff_cnn/sunwoo_siam/siam_CNN/jetson_nano/mitbih_database/'

# checkpoint
checkpoint_name = 'test.pth'
device = torch.device('cuda:7')

patients = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]

k = 20
n_segments = k
std_dist = 0.1 # std 구하기
std_inter = 100 # std 구하기

dists_n = []
inters_n = []
dists_s = []
inters_s = []
dists_v = []
inters_v = []
dists_f = []
inters_f = []
dists_q = []
inters_q = []


for idx, patient in enumerate(patients):
    print(patient)
    sig, annotations = load(dataset_path, patient, from_npy=False) # 각 점마다의 time, type, sample#
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
    
    normal_segments = []
    dists = []
    inters = []
    peaks = []


    for i in range(0,k) :
        type = annotations[i]['type']
        
        if type == 0 :
            segment = X[i]
            normal_segments.append(segment)
        else :
            continue
    segment_avg = np.average(normal_segments, axis=0)

    for i in range(1,k) :
        #print(i)
        type = annotations[i]['type']
        ann_time_pre = annotations[i-1]['time']
        ann_time = annotations[i]['time']
        time_s_pre = float(ann_time_pre[-3:])*0.001 + float(ann_time_pre[-6:-4]) + float(float(ann_time_pre[:-7])/60)
        time_s = float(ann_time[-3:])*0.001 + float(ann_time[-6:-4]) + float(float(ann_time[:-7])/60) # s로 통일
        inter = time_s - time_s_pre
        if type == 0 :
            #print('normal')
            #print(peaks)
            segment = X[i]
            l = len(segment)
            dist = modified_dtw(segment_avg[:l//2], segment_avg[l//2:], segment[:l//2], segment[l//2:])
            dists.append(dist)
            inters.append(inter)
        else :
            continue

    dist_avg = np.average(dists)
    inter_avg = np.average(inters)


    for i in range(k,len(X)):
        type = annotations[i]['type']
        ann_time_pre = annotations[i-1]['time']
        ann_time = annotations[i]['time']
        #print(ann_time)
        time_s_pre = float(ann_time_pre[-3:])*0.001 + float(ann_time_pre[-6:-4]) + float(float(ann_time_pre[:-7])/60)
        time_s = float(ann_time[-3:])*0.001 + float(ann_time[-6:-4]) + float(float(ann_time[:-7])/60) # s로 통일
        inter = time_s - time_s_pre
        #print(peaks)
        segment = X[i]
        l = len(segment)
        dist = modified_dtw(segment_avg[:l//2], segment_avg[l//2:], segment[:l//2], segment[l//2:])


        if type == 0 : 
            dists_n.append(dist)
            inters_n.append(inter)
            segment_avg = segment_avg * 0.9 + segment * 0.1 # update

        elif type == 1 :
            dists_s.append(dist)
            inters_s.append(inter)

        elif type == 2 :
            dists_v.append(dist)
            inters_v.append(inter)

        elif type == 3 :
            dists_f.append(dist)
            inters_f.append(inter)

        elif type == 4 :
            dists_q.append(dist)
            inters_q.append(inter)
        
        else :
            continue

        assert len(dists_n) == len(inters_n)
        assert len(dists_s) == len(inters_s)
        assert len(dists_v) == len(inters_v)
        assert len(dists_f) == len(inters_f)
        assert len(dists_q) == len(inters_q)


plt.hist(dists_n, color = 'green', alpha = 0.2, bins = 50,range = [0,2], label = 'N', density = True)
plt.hist(dists_s, color = 'red', alpha = 0.2, bins = 50, range = [0,2], label = 'S', density = True)
plt.hist(dists_v, color = 'blue', alpha = 0.2, bins = 50, range = [0,2], label = 'V', density = True)
plt.hist(dists_f, color = 'yellow', alpha = 0.2, bins = 50, range = [0,2], label = 'F', density = True)
plt.hist(dists_q, color = 'gray', alpha = 0.2, bins = 50, range = [0,2], label = 'Q', density = True)
#plt.hist(inters_n, color = 'green', alpha = 0.2, bins = 50, range = [0.2,1.5],label = 'N', density = True)
#plt.hist(inters_s, color = 'red', alpha = 0.2, bins = 50, range = [0.2,1.5],label = 'S', density = True)
#plt.hist(inters_v, color = 'blue', alpha = 0.2, bins = 50, range = [0.2,1.5],label = 'V', density = True)
#plt.hist(inters_f, color = 'yellow', alpha = 0.2, bins = 50, range = [0.2,1.5],label = 'F', density = True)
#plt.hist(inters_q, color = 'gray', alpha = 0.2, bins = 50, range =[0.2,1.5],label = 'Q', density = True)
plt.legend()
plt.show()
