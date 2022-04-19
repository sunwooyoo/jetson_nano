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

class BaseDataset(Dataset):
    def __init__(self, path, dataset='ds1', use_detector=False, from_npy=False): # norm='before'
        self.path = path
        self.segmentation = get_segmentation('windowing')#('dynamic_center')
        self.from_npy = from_npy
        if from_npy:
            self.preproc = get_preproc('none')
        else:
            self.preproc = partial(get_preproc('bandpass'), fc=(1.5, 150))
        self.zscore = True
        self.dataset = dataset
        self.patients = BaseDataset.get_patients(dataset)
        self.detector = None if not use_detector else AnomalyDetector()
        self.use_detector = use_detector
        #self.norm = norm # normalization 순서
        self.confusion_matrix = np.zeros((5, 5), dtype=int)
        self.labels = []
        self.preds = []
        self.make_dataset()
    
    def make_dataset(self):
        self.X = []
        self.Y = []
        for idx, patient in enumerate(self.patients):
            X, annotations = self.make_samples(patient)
            X, Y = self.sample_reduction(X, annotations, self.use_detector)
            self.X += X
            self.Y += Y
            sys.stdout.write('\r loading {}: {}/{}'.format(
                self.dataset, idx+1, len(self.patients)
            ))
        sys.stdout.write('\r dataset {} is loaded\n'.format(self.dataset))
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def make_samples(self, patient):
        sig, annotations = BaseDataset.load(self.path, patient, from_npy=self.from_npy) # bandpass
        sig_f = self.preproc(sig)
        if self.use_detector:
            annotations = self.anomaly_detection(sig=sig, annotations=annotations)
        segments, annotations = self.segmentation(sig_f, annotations) # segmentation 
        X = []
        for segment in segments:
            if self.zscore:
                segment = (segment - segment.min())/(segment.max()-segment.min())
            X.append(segment)
        assert len(X) == len(annotations)
        return X, annotations

    def sample_reduction(self, X: List[Any], annotations: Dict[str, Any], use_detector: bool)->Tuple[List[Any], List[int]]:
        if use_detector:
            normal_preds = []
            i = 0
            while i < len(annotations):
                annotation = annotations[i]
                truth, pred = annotation['type'], annotation['pred']
                truth2 = 0 if truth == 0 else 1
                if truth2 == 0 and pred == 'N':
                    # remove true negative
                    normal_preds.append([0,0])
                    annotations.pop(i)
                    X.pop(i)
                else:
                    i += 1
            labels, preds = map(list, zip(*normal_preds))
            self.confusion_matrix += confusion_matrix(labels,preds, labels = [0,1,2,3,4])
            self.labels += labels
            self.preds += preds
        return X, [ann['type'] for ann in annotations]

    def anomaly_detection(self, mode='online', sig=None, annotations=None, segments=None):
        if mode == 'offline':
            # load annotation file containing prediction results
            raise NotImplementedError()
        elif mode == 'online':
            # detect
            if annotations is None:
                raise ValueError()
            if segments is None and sig is None:
                raise ValueError()
            elif sig is None:
                return self.detector.detect(segments, annotations)
            elif segments is None:
                return self.detector.detect_from_signal(sig, annotations)
            else:
                raise ValueError()
        else:
            raise NotImplementedError()
    
    @staticmethod
    def load(path, patient, remove_nonaami=True, from_npy=False):
        normal_beats = ['N', 'L', 'R', 'e', 'j']
        sup_beats = ['A', 'a', 'J', 'S']
        ven_beats = ['V', 'E']
        fusion_beats = ['F']
        unknown_beats = ['/', 'f', 'Q']
        aami_beats = normal_beats+sup_beats+ven_beats+fusion_beats+unknown_beats
        csv_path = path+str(patient)+'.csv'
        ann_path = path+str(patient)+'annotations.txt'
        if from_npy:
            npy_path = path+'{}_bandpass.npy'.format(patient)
            sig = np.load(npy_path)
        else:
            df = pd.read_csv(csv_path)
            sig = df["'MLII'"].to_numpy()
        annotations = []
        with open(ann_path, 'r') as f:
            next(f)
            for line in f:
                ann_split = line.split()
                type = ann_split[2]
                if type not in aami_beats and remove_nonaami:
                    continue
                if type in normal_beats:
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

        return sig, annotations

    @staticmethod
    def get_patients(dataset):
        DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
        DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
        DS11 = [101, 106, 108, 109, 114, 115, 116, 119, 122, 209, 223] # for training
        DS12 = [112, 118, 124, 201, 203, 205, 207, 208, 215, 220, 230] # for validation
        if dataset == 'ds1':
            patients = DS1
        elif dataset == 'ds2':
            patients = DS2
        elif dataset == 'ds11':
            patients = DS11
        elif dataset == 'ds12' :
            patients = DS12
        elif dataset == 'all':
            patients = DS1+DS2
        else:
            patients = [210]
        return patients

class DatasetWithRR(BaseDataset):
    # DatasetWithFullAverager로 대체
    def __init__(self, path, n_intervals, dataset='ds1', use_detector=False, from_npy=False):
        self.n_intervals = n_intervals
        return super().__init__(path, dataset=dataset, use_detector=use_detector, from_npy=from_npy)
    
    def make_dataset(self):
        self.X = []
        self.Y = []
        for idx, patient in enumerate(self.patients):
            segments, annotations = self.make_samples(patient)
            intervals = [0]
            for ann_pre, ann_cur in zip(annotations, annotations[1:]):
                r_pre, r_cur = ann_pre['Sample#'], ann_cur['Sample#']
                intervals.append(r_cur-r_pre)
            X = []
            Y = []
            for i in range(self.n_intervals, len(segments)):
                interval_seq = np.array(intervals[i-self.n_intervals+1:i+1])
                # min-max scaling
                # interval_seq = (interval_seq-interval_seq.min())/(interval_seq.max()-interval_seq.min())
                # scaling with average
                # interval_seq = interval_seq/interval_seq.mean()
                interval_seq = interval_seq/interval_seq[-1]
                X.append([segments[i], interval_seq])
                Y.append(annotations[i])
            X, Y = self.sample_reduction(X, Y, self.use_detector)
            self.X += X
            self.Y += Y
            sys.stdout.write('\r loading {}: {}/{}'.format(
                self.dataset, idx+1, len(self.patients)
            ))
        sys.stdout.write('\r dataset {} is loaded\n'.format(self.dataset))

class DatasetWithFullAverager(BaseDataset):
    '''
    모든 beat에 대한 average signal, rr interval
    [x, x_avg, intervals, intervals_avg], y = dset[:]
    '''
    def __init__(self, path, n_intervals, dataset='ds1', use_detector=False, average_size=10,from_npy=False):
        self.n_intervals = n_intervals
        self.average_size = average_size
        super().__init__(path, dataset=dataset, use_detector=use_detector, from_npy=from_npy)
    
    def make_dataset(self):
        self.X = []
        self.Y = []
        for idx, patient in enumerate(self.patients):
            sys.stdout.write('\r loading {}: {}/{}'.format(
                self.dataset, idx+1, len(self.patients)
            ))
            segments, annotations = self.make_samples(patient)
            intervals = [0]
            for ann_pre, ann_cur in zip(annotations, annotations[1:]):
                r_pre, r_cur = ann_pre['Sample#'], ann_cur['Sample#']
                intervals.append(r_cur-r_pre)
            buffer_segments = []
            buffer_intervals = []
            X = []
            Y = []
            for i in range(len(segments)):
                assert len(buffer_intervals) == len(buffer_segments)
                if len(buffer_segments) == self.average_size:
                    x = segments[i]
                    x_avg = np.average(buffer_segments, axis=0)
                    rr = np.array(buffer_intervals[-self.n_intervals+1:]+[intervals[i]])
                    rr = rr/rr[-1]
                    rr_avg = np.array(buffer_intervals[-self.n_intervals:])
                    rr_avg = rr_avg/rr_avg[-1]
                    X.append([x, x_avg, rr, rr_avg])
                    Y.append(annotations[i])
                buffer_segments.append(segments[i])
                buffer_intervals.append(intervals[i])
                if len(buffer_segments) > self.average_size:
                    buffer_segments.pop(0)
                    buffer_intervals.pop(0)
            X, Y = self.sample_reduction(X, Y, self.use_detector)
            self.X += X
            self.Y += Y
        sys.stdout.write('\r dataset {} is loaded\n'.format(self.dataset))

class DatasetWithNormalAverager(BaseDataset):
    '''
    true normal beat에 대한 average signal, rr interval return
    [x, x_avg, intervals, intervals_avg], y = dset[:]
    '''
    def __init__(self, path, n_intervals, average_size=10, dataset='ds1', use_detector=False, from_npy=False):
        self.n_intervals = n_intervals
        self.average_size = average_size
        super().__init__(path, dataset=dataset, use_detector=use_detector, from_npy=from_npy)
    
    def make_dataset(self):
        self.X = []
        self.Y = []
        for idx, patient in enumerate(self.patients):
            segments, annotations = self.make_samples(patient)
            intervals = [0]
            for ann_pre, ann_cur in zip(annotations, annotations[1:]):
                r_pre, r_cur = ann_pre['Sample#'], ann_cur['Sample#']
                intervals.append(r_cur-r_pre)
            buffer_intervals = []
            buffer_segments = []
            X = []
            Y = []
            for i in range(len(segments)):
                assert len(buffer_intervals) == len(buffer_segments)
                if len(buffer_segments) == self.average_size:
                    x = segments[i]
                    x_avg = np.average(buffer_segments, axis=0)
                    rr = np.array(buffer_intervals[-self.n_intervals+1:]+[intervals[i]])
                    rr = rr/rr[-1]
                    rr_avg = np.array(buffer_intervals[-self.n_intervals:])
                    rr_avg = rr_avg/rr_avg[-1]
                    X.append([x, x_avg, rr, rr_avg])
                    Y.append(annotations[i])
                type = annotations[i]['type']
                if type == 0:
                    buffer_segments.append(segments[i])
                    buffer_intervals.append(intervals[i])
                    if len(buffer_segments) > self.average_size:
                        buffer_segments.pop(0)
                        buffer_intervals.pop(0)
            X, Y = self.sample_reduction(X, Y, self.use_detector)
            self.X += X
            self.Y += Y
            sys.stdout.write('\r loading {}: {}/{}'.format(
                self.dataset, idx+1, len(self.patients)
            ))
        sys.stdout.write('\r dataset {} is loaded\n'.format(self.dataset))

    def sample_reduction(self, X: List[Any], annotations: Dict[str, Any], use_detector: bool) -> Tuple[List[Any], List[int]]:
        if use_detector:
            normal_preds = []
            i = 0
            while i < len(annotations):
                annotation = annotations[i]
                truth, pred = annotation['type'], annotation['pred']
                truth2 = 0 if truth == 0 else 1
                if truth2 == 0 and pred == 'N':
                    normal_preds.append([truth, 0])
                    # remove true negative
                    annotations.pop(i)
                    X.pop(i)
                else:
                    i += 1
            labels, preds = map(list, zip(*normal_preds))
            self.confusion_matrix += confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
            self.labels += labels
            self.preds += preds
        return X, [ann['type'] for ann in annotations]

class DatasetWithDetector(BaseDataset):
    '''
    detector의 normal prediction에 대한 average signal, rr interval return
    [x, x_avg, intervals, intervals_avg], y = dset[:]
    '''
    def __init__(self, path, n_intervals, average_size=10, dataset='ds1', return_normal=True, from_npy=False):
        self.n_intervals = n_intervals
        self.average_size = average_size
        self.return_normal = return_normal
        super().__init__(path, dataset=dataset, use_detector=True, from_npy=from_npy)
    
    def make_dataset(self):
        self.X = []
        self.Y = []
        for idx, patient in enumerate(self.patients):
            sys.stdout.write('\r loading {}: {}/{}'.format(
                self.dataset, idx+1, len(self.patients)
            ))
            segments, annotations = self.make_samples(patient)
            intervals = [0]
            for ann_pre, ann_cur in zip(annotations, annotations[1:]):
                r_pre, r_cur = ann_pre['Sample#'], ann_cur['Sample#']
                intervals.append(r_cur-r_pre)
            buffer_segments = []
            buffer_intervals = []
            X = []
            Y = []
            for i in range(len(segments)):
                assert len(buffer_intervals) == len(buffer_segments)
                if len(buffer_segments) == self.average_size:
                    x = segments[i]
                    x_avg = np.average(buffer_segments, axis=0)
                    rr = np.array(buffer_intervals[-self.n_intervals+1:]+[intervals[i]])
                    rr = rr/rr[-1]
                    rr_avg = np.array(buffer_intervals[-self.n_intervals:])
                    rr_avg = rr_avg/rr_avg[-1]
                    X.append([x, x_avg, rr, rr_avg])
                    Y.append(annotations[i])
                pred = annotations[i]['pred']
                if pred == 'N':
                    buffer_segments.append(segments[i])
                    buffer_intervals.append(intervals[i])
                    if len(buffer_segments) > self.average_size:
                        buffer_segments.pop(0)
                        buffer_intervals.pop(0)
            X, Y = self.sample_reduction(X, Y, not self.return_normal)
            self.X += X
            self.Y += Y
        sys.stdout.write('\r dataset {} is loaded\n'.format(self.dataset))
    
    def sample_reduction(self, X: List[Any], annotations: Dict[str, Any], use_detector: bool) -> Tuple[List[Any], List[int]]:
        '''
        remove negative samples,
        update confusion matrix
        '''
        if use_detector:
            normal_preds = [[0, 0]]
            i = 0
            while i < len(annotations):
                annotation = annotations[i]
                truth, pred = annotation['type'], annotation['pred']
                if pred == 'N':
                    normal_preds.append([truth, 0])
                    annotations.pop(i)
                    X.pop(i)
                else:
                    i += 1
            labels, preds = map(list, zip(*normal_preds))
            self.confusion_matrix += confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
            self.labels += labels
            self.preds += preds
        return X, [ann['type'] for ann in annotations]


class DatasetWithEnhancedDetector(DatasetWithDetector):
    def __init__(self, path, n_intervals, average_size=10, dataset='ds1', return_normal=True):
        super().__init__(path, n_intervals, average_size=average_size, dataset=dataset, return_normal=return_normal)
    
    def make_samples(self, patient):
        sig, annotations = BaseDataset.load(self.path, patient)
        sig_f = self.preproc(sig)
        segments, annotations = self.segmentation(sig_f, annotations)
        annotation = self.anomaly_detection(segments=segments, annotations=annotations)
        X = []
        for segment in segments:
            if self.zscore:
                segment = (segment - segment.min())/(segment.max()-segment.min())
            X.append(segment)
        assert len(X) == len(annotations)
        return X, annotations
    
    def make_dataset(self):
        self.detector = EnhandcedAnomalyDetector()
        return super().make_dataset()
