from abc import *
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import Dataset
from preprocessing import get_preproc, get_segmentation
from detector import AnomalyDetector, EnhandcedAnomalyDetector
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from functools import reduce
from dtaidistance import dtw

def load(path, patient, remove_nonaami=True, from_npy=False):
        normal_beats = ['N', 'L', 'R', 'e', 'j']
        sup_beats = ['A', 'a', 'J', 'S']
        ven_beats = ['V', 'E']
        fusion_beats = ['F']
        unknown_beats = ['/', 'f', 'Q']
        aami_beats = normal_beats+sup_beats+ven_beats+fusion_beats+unknown_beats
        csv_path = path+str(patient)+'.csv'
        ann_path = path+str(patient)+'annotations.txt'
        
        df = pd.read_csv(csv_path)
        sig = df["'MLII'"].to_numpy()
            
        annotations = []
        with open(ann_path, 'r') as f:
            next(f)
            for line in f:
                ann_split = line.split()
                #print(ann_split)
                type = ann_split[2]
                if type not in aami_beats:
                    #print('no')
                    continue
                if type in normal_beats:
                    #print('normal')
                    type = 0
                elif type in sup_beats:
                    type = 1
                elif type in ven_beats:
                    type = 2
                elif type in fusion_beats:
                    type = 3
                else:
                    type = 4
                annotations.append({'time': ann_split[0], 'Sample#': int(ann_split[1]), 'type': type})
                #print(annotations)
        return sig, annotations


def load_for_original(path, patient, remove_nonaami=True, from_npy=False):
        normal_beats = ['N', 'L', 'R', 'e', 'j']
        sup_beats = ['A', 'a', 'J', 'S']
        ven_beats = ['V', 'E']
        fusion_beats = ['F']
        unknown_beats = ['/', 'f', 'Q']
        aami_beats = normal_beats+sup_beats+ven_beats+fusion_beats+unknown_beats
        csv_path = path+str(patient)+'.csv'
        ann_path = path+str(patient)+'annotations.txt'
        
        df = pd.read_csv(csv_path)
        sig = df["'MLII'"].to_numpy()
            
        annotations = []
        with open(ann_path, 'r') as f:
            next(f)
            for line in f:
                ann_split = line.split()
                #print(ann_split)
                type = ann_split[2]
                if type not in aami_beats:
                    #print('no')
                    continue
                if type in normal_beats:
                    #print('normal')
                    type = 0
                elif type in sup_beats:
                    type = 1
                elif type in ven_beats:
                    type = 2
                elif type in fusion_beats:
                    type = 3
                else:
                    type = 4
                annotations.append({'time': ann_split[0], 'Sample#': int(ann_split[1]), 'type': type, 'pred': ann_split[3]})
                #print(annotations)
        return sig, annotations



def modified_dtw(xl, xr, yl, yr):
    return dtw.distance_fast(xl, yl) + dtw.distance_fast(xr, yr)

def approx_exp_moving_avg(avg_pre, sample_new, w):
    if avg_pre.shape != sample_new.shape:
        return avg_pre
    else:
        avg = sample_new/w + (1-1/w)*avg_pre
        return avg

def compute_binary_accuracy(num_tn, num_tp, num_fn, num_fp):
    if (num_tp + num_fn) == 0:
        se = 0
    else:
        se = num_tp/(num_tp+num_fn)
    sp = num_tn/(num_tn+num_fp)
    acc = (num_tp+num_tn)/(num_tp+num_tn+num_fn+num_fp)
    return se, sp, acc
