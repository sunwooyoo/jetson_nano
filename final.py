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
from detector_utils import load, modified_dtw, approx_exp_moving_avg, compute_binary_accuracy
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import SiameseNetwork_ete
from loss import ContrastiveLoss
from dataset import DatasetWithNormalAverager,DatasetWithDetector
from sklearn.metrics import confusion_matrix, classification_report
from metrics import AvgLoss, Accuracy, Sensitivity, Specificity, print_metrics, reset_metrics

font2 = {'family': 'Times New Roman',
        'color':  'blue',
        'weight': 'bold',
        'size': 12,
        'alpha': 0.7}
box1 = {'boxstyle': 'round',
        'ec': (1.0, 0.5, 0.5),
        'fc': (1.0, 0.8, 0.8)}
max_epochs = 100
batch_size = 256
n_intervals = 10
dataset_path = '/home/swyoo/diff_cnn/sunwoo_siam/siam_CNN/jetson_nano/mitbih_database/'

# checkpoint
checkpoint_name = 'test.pth'
device = torch.device('cuda:7')

patients = [ 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

k = 10
n_segments = k
std_dist = 0.2 # std 구하기
std_inter = 0.2 # std 구하기
tp = 0
tn = 0
fp = 0
fn = 0

model = SiameseNetwork_ete() # define model 
model.to(device)


def validate(latency,  seg_1, x, x_avg, rr, rr_avg):
    model.eval()
    start = time.time()
    with torch.no_grad():
        batch_size = 1
        x = x.to(dtype=torch.float32, device=device).view(batch_size, 1, -1)
        x_avg = x_avg.to(dtype=torch.float32, device=device).view(batch_size, 1, -1)
        rr = rr.to(dtype=torch.float32, device=device)
        rr_avg = rr_avg.to(dtype=torch.float32, device=device)
        out1, out2, out_center, out = model(x, x_avg, rr, rr_avg) # out1 out2 for contrastive loss
        preds = torch.argmax(out, dim=1)
        print(preds)
        latency2 = time.time() - start
        t = np.arange(0,len(seg_1),1)
        plt.plot(t,seg_1, color = 'indianred', label = 'segment')
        plt.title('prediction label : '+ str(preds))
        plt.text(2.0, 0.0, str(latency+ latency2)+'ms', fontdict=font2, bbox=box1)
        #plt.text(-10.0, 0.08, str(power1), fontdict=font3, bbox=box2)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    return preds


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

    #### initialization ####
    peaks = []
    normal_segments = []
    dists = []
    inters = []
    length = 10
    pre_dists = []
    pre_inters = []
    avg_inters = []

    init_normal = []
    init_dists = []
    init_inters = []

    for i in range(1,k) :
        segment = X[i]
        normal_segments.append(segment)
    segment_avg = np.average(normal_segments, axis=0)
    seg_1 = segment_avg

    
    for i in range(1,k):
        type = annotations[i]['type']
        #print(type)
        ann_time_pre = annotations[i-1]['time']
        ann_time = annotations[i]['time']
        time_s_pre = float(ann_time_pre[-3:])*0.001 + float(ann_time_pre[-6:-4]) + float(float(ann_time_pre[:-7])/60)
        time_s = float(ann_time[-3:])*0.001 + float(ann_time[-6:-4]) + float(float(ann_time[:-7])/60) # s로 통일
        inter = time_s - time_s_pre
        #print(peaks)
        segment = X[i]
        l = len(segment)
        dist = modified_dtw(segment_avg[:l//2], segment_avg[l//2:], segment[:l//2], segment[l//2:])
        #dists.append(dist)
        
        if dist <= 0.5 or inter >= 0.62 : 
            
            #print('seleted')
            
            dists.append(dist)
            if len(dists) > length:
                dists.pop(0)
            inters.append(inter)
            if len(inters) > length:
                inters.pop(0)
            avg_inters.append(inter)
            if len(avg_inters) > length:
                avg_inters.pop(0)

            init_normal.append(segment)
        else:
            continue

    # final init value
    segment_avg = np.average(init_normal, axis=0)
    dist_avg = np.average(dists)
    inter_avg = np.average(avg_inters)
    #print("filtering time :", time.time() - start)
    #t = np.arange(0,len(seg_1),1)
    #plt.plot(t,seg_1, color = 'powderblue')
    #plt.plot(t,segment_avg,color='indianred')
    #plt.show()
    trues = []
    preds = []
    seg_anoaml = []

    for i in range(k,len(X)):
        start = time.time()
        type = annotations[i]['type']
        trues.append(type) # for evaluate

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
        
        score_dist = np.abs(dist-dist_avg)-std_dist
        score_inter = np.abs(inter-inter_avg)-std_inter
        
        #print(torch.Tensor(segment).view(1,-1).shape)
        #print(torch.Tensor(segment_avg).view(1,-1).shape)
        #print(torch.Tensor(inters).view(1,-1).shape)
        #print(torch.Tensor(avg_inters).view(1,-1).shape)


        if score_dist < 0 and score_inter < 0:

            segment_avg = segment_avg * 0.9 + segment * 0.1
            
            dists.append(dist)
            if len(dists) > length:
                dists.pop(0)
            inters.append(inter)
            if len(inters) > length:
                inters.pop(0)
            avg_inters.append(inter)
            if len(avg_inters) > length:
                avg_inters.pop(0)
            dist_avg = np.average(dists)
            inter_avg = np.average(avg_inters)    

            latency = time.time() - start

            t = np.arange(0,len(segment),1)
            plt.plot(t,segment, color = 'indianred', label = 'segment')
            plt.title('prediction label : Normal ')
            plt.text(2.0, 0.0, str(latency)+'ms', fontdict=font2, bbox=box1)
            #plt.text(-10.0, 0.08, str(power1), fontdict=font3, bbox=box2)
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()



            preds.append('N')  # for evaluate
            # jetson에서는 여기서 power 측정하고 time 측정하고 결과 보여주기

        else : 
            inters.append(inter)
            if len(inters) > length:
                inters.pop(0)
            preds.append('A')  # for evaluate
            x = torch.Tensor(segment).view(1,-1)
            x_avg = torch.Tensor(segment_avg).view(1,-1)
            rr = torch.Tensor(inters).view(1,-1)
            rr_avg = torch.Tensor(avg_inters).view(1,-1)
            model.load_state_dict(torch.load('/home/swyoo/diff_cnn/sunwoo_siam/siam_CNN/jetson_nano/test26.pth'))
            latency = time.time() - start
            validate(latency, segment, x, x_avg, rr, rr_avg)
            # x, x_avg, rr, rr_avg, 
            # torch.Size([64, 279])
            # torch.Size([64, 10])




    #print(len(trues))
    #print(len(preds))
    # evaluate
    '''
    for pred, truth in zip(preds, trues):

        if truth == 0:

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

print(tn, fn, fp, tp)
se, sp, acc = compute_binary_accuracy(tn, tp, fn, fp)
print(se, sp, acc)

'''

