
import numpy as np
import torch
from functools import reduce
from dtaidistance import dtw
from preprocessing import get_preproc, get_segmentation

def modified_dtw(xl, xr, yl, yr):
    return dtw.distance_fast(xl, yl) + dtw.distance_fast(xr, yr)

def approx_exp_moving_avg(avg_pre, sample_new, w):
    if avg_pre.shape != sample_new.shape:
        return avg_pre
    else:
        avg = sample_new/w + (1-1/w)*avg_pre
        return avg

class AnomalyDetector():
    def __init__(
        self, 
        idx_train_start=3,
        idx_train_end=20,
        n_dists=10,
        n_inters=40,
        n_segments=10,
        std_dist=0.1,
        std_inter=100
    ):
        self.idx_train_start = idx_train_start
        self.idx_train_end = idx_train_end
        self.n_dists = n_dists
        self.n_inters = n_inters
        self.n_segments = n_segments
        self.std_dist = std_dist
        self.std_inter = std_inter
        self.history = None
    
    def detect(self, segments, annotations):
        self.history = {
            'dist':[0 for _ in range(self.idx_train_end)],
            'inter':[0 for _ in range(self.idx_train_end)],
            'score_dist':[0 for _ in range(self.idx_train_end)],
            'score_inter':[0 for _ in range(self.idx_train_end)],
            'dist_avg':[0 for _ in range(self.idx_train_end)],
            'inter_avg':[0 for _ in range(self.idx_train_end)]
        }
        len_annotations = len(annotations)
        # R peak가 중앙에 있어야함.
        peaks = reduce(lambda acc, ann: acc + [ann['Sample#']], annotations[1:], [annotations[0]['Sample#']])
        dists = []
        inters = []
        preds = ['T' for _ in range(self.idx_train_end)]
        normal_segments = []

        #### initialization ####
        assert self.idx_train_start >= 1
        for i in range(self.idx_train_start, self.idx_train_end):
            segment = segments[i]
            normal_segments.append(segment)
            if len(normal_segments) > self.n_segments:
                normal_segments.pop(0)
            segment_avg = np.average(normal_segments, axis=0)
        
        for i in range(self.idx_train_start, self.idx_train_end):
            segment = segments[i]
            l = len(segment)
            dist = modified_dtw(segment_avg[:l//2], segment_avg[l//2:], segment[:l//2], segment[l//2:])
            inter = peaks[i] - peaks[i-1]
            dists.append(dist)
            if len(dists) > self.n_dists:
                dists.pop(0)
            inters.append(inter)
            if len(inters) > self.n_inters:
                inters.pop(0)
            dist_avg = np.average(dists)
            inter_avg = np.average(inters)

        #### detection phase ####
        for i in range(self.idx_train_end, len(peaks)):
            segment = segments[i]
            l = len(segment)
            dist = modified_dtw(segment_avg[:l//2], segment_avg[l//2:], segment[:l//2], segment[l//2:])
            inter = peaks[i] - peaks[i-1]
            score_dist = np.abs(dist-dist_avg)-self.std_dist
            score_inter = np.abs(inter-inter_avg)-self.std_inter

            if score_dist < 0 and score_inter < 0:
                preds.append('N')
                normal_segments.append(segment)
                if len(normal_segments) > self.n_segments:
                    normal_segments.pop(0)
                dists.append(dist)
                if len(dists) > self.n_dists:
                    dists.pop(0)
                inters.append(inter)
                if len(inters) > self.n_inters:
                    inters.pop(0)
            else:
                preds.append('A')
            dist_avg = np.average(dists)
            inter_avg = np.average(inters)
            segment_avg = np.average(normal_segments, axis=0)
            self.history['dist'].append(dist)
            self.history['inter'].append(inter)
            self.history['score_dist'].append(score_dist)
            self.history['score_inter'].append(score_inter)
            self.history['dist_avg'].append(dist_avg)
            self.history['inter_avg'].append(inter_avg)
        
        assert len(preds)==len(annotations)
        for pred, annotation in zip(preds, annotations):
            annotation['pred'] = pred
        assert len_annotations == len(annotations)
        return annotations
    
    def detect_from_signal(self, sig, annotations):
        # segmentation, preprocessing 따로 구현하기 위한 함수
        #preprocessing = get_preproc('bandpass')
        #sig = preprocessing((sig-1024)/1024, fc=(1.5, 150))
        segmentation = get_segmentation('windowing')
        segments, annotations = segmentation(sig, annotations, window=(140, 140))
        assert len(segments[0]) == 280
        annotations = self.detect(segments, annotations)
        return annotations
    
    def evaluate(self, sig=None, segments=None, annotations=None):
        #if annotations is None:
        #    raise ValueError()
        #if segments is None and sig is None:
        #    raise ValueError()
        #elif segments is None:
        #    annotations = self.detect_from_signal(sig, annotations)
        #else:
        #    annotations = self.detect(segments, annotations)
        tp = []
        tn = []
        fp = []
        fn = []
        for idx, annotation in enumerate(annotations):
            truth = annotation['type']
            pred = annotation['pred']
            if truth == 0:
                if pred == 'N':
                    # true negative
                    tn.append(idx)
                elif pred == 'A':
                    # false positive
                    fp.append(idx)
            else:
                assert truth is not None
                if pred == 'N':
                    # false negative
                    fn.append(idx)
                elif pred == 'A':
                    # true positive
                    tp.append(idx)
        self.history['tp'] = tp
        self.history['tn'] = tn
        self.history['fp'] = fp
        self.history['fn'] = fn
        se, sp, acc = AnomalyDetector.compute_binary_accuracy(len(tn), len(tp), len(fn), len(fp))
        print(len(tn), len(tp), len(fn), len(fp))
        print(se, sp, acc)
        return annotations

    
    @staticmethod
    def compute_binary_accuracy(num_tn, num_tp, num_fn, num_fp):
        if (num_tp + num_fn) == 0:
            se = 0
        else:
            se = num_tp/(num_tp+num_fn)
        sp = num_tn/(num_tn+num_fp)
        acc = (num_tp+num_tn)/(num_tp+num_tn+num_fn+num_fp)
        return se, sp, acc

class EnhandcedAnomalyDetector(AnomalyDetector):
    '''
    classifier의 결과를 사용하는 detector
    '''
    def __init__(self, clf, device, thr_clf=0.5, n_intervals_clf=10, idx_train_start=3, idx_train_end=20, n_dists=10, n_inters=40, n_segments=10, std_dist=0.1, std_inter=100):
        super().__init__(idx_train_start=idx_train_start, idx_train_end=idx_train_end, n_dists=n_dists, n_inters=n_inters, n_segments=n_segments, std_dist=std_dist, std_inter=std_inter)
        self.n_intervals_clf = n_intervals_clf
        self.clf = clf
        self.device = device
        self.thr_clf = thr_clf
        self.clf.eval()
        self.abnormal = []
    
    def detect(self, segments, annotations):
        self.history = {
            'dist':[0 for _ in range(self.idx_train_end)],
            'inter':[0 for _ in range(self.idx_train_end)],
            'score_dist':[0 for _ in range(self.idx_train_end)],
            'score_inter':[0 for _ in range(self.idx_train_end)],
            'dist_avg':[0 for _ in range(self.idx_train_end)],
            'inter_avg':[0 for _ in range(self.idx_train_end)]
        }
        len_annotations = len(annotations)
        # R peak가 중앙에 있어야함.
        peaks = reduce(lambda acc, ann: acc + [ann['Sample#']], annotations[1:], [annotations[0]['Sample#']])
        dists = []
        inters = []
        preds = ['T' for _ in range(self.idx_train_end)]
        normal_segments = []

        #### initialization ####
        assert self.idx_train_start >= 1
        for i in range(self.idx_train_start, self.idx_train_end):
            segment = segments[i]
            normal_segments.append(segment)
            if len(normal_segments) > self.n_segments:
                normal_segments.pop(0)
            segment_avg = np.average(normal_segments, axis=0)
        
        for i in range(self.idx_train_start, self.idx_train_end):
            segment = segments[i]
            l = len(segment)
            dist = modified_dtw(segment_avg[l//2-50:l//2], segment_avg[l//2:l//2+50], segment[l//2-50:l//2], segment[l//2:l//2+50])
            inter = peaks[i] - peaks[i-1]
            dists.append(dist)
            if len(dists) > self.n_dists:
                dists.pop(0)
            inters.append(inter)
            if len(inters) > self.n_inters:
                inters.pop(0)
            dist_avg = np.average(dists)
            inter_avg = np.average(inters)

        #### detection phase ####
        for i in range(self.idx_train_end, len(peaks)):
            segment = segments[i]
            l = len(segment)
            dist = modified_dtw(segment_avg[l//2-50:l//2], segment_avg[l//2:l//2+50], segment[l//2-50:l//2], segment[l//2:l//2+50])
            inter = peaks[i] - peaks[i-1]
            score_dist = np.abs(dist-dist_avg)-self.std_dist
            score_inter = np.abs(inter-inter_avg)-self.std_inter

            if score_dist < 0 and score_inter < 0:
                preds.append('N')
                normal_segments.append(segment)
                if len(normal_segments) > self.n_segments:
                    normal_segments.pop(0)
                dists.append(dist)
                if len(dists) > self.n_dists:
                    dists.pop(0)
                inters.append(inter)
                if len(inters) > self.n_inters:
                    inters.pop(0)
            else:
                # check classifier result
                is_normal = self.forward_clf(segment, segment_avg, inter, inters)
                if is_normal:
                    preds.append('A')
                    # prediction은 그대로 A, 평균계산에는 포함시키기
                    segment_avg = approx_exp_moving_avg(segment_avg, segment, self.n_segments)
                    dists.append(dist)
                    if len(dists) > self.n_dists:
                        dists.pop(0)
                    inters.append(inter)
                    if len(inters) > self.n_inters:
                        inters.pop(0)
                else:
                    preds.append('A')
            dist_avg = np.average(dists)
            inter_avg = np.average(inters)
            segment_avg = np.average(normal_segments, axis=0)
            self.history['dist'].append(dist)
            self.history['inter'].append(inter)
            self.history['score_dist'].append(score_dist)
            self.history['score_inter'].append(score_inter)
            self.history['dist_avg'].append(dist_avg)
            self.history['inter_avg'].append(inter_avg)
        
        assert len(preds)==len(annotations)
        for pred, annotation in zip(preds, annotations):
            annotation['pred'] = pred
        assert len_annotations == len(annotations)
        return annotations
    
    def detect_from_signal(self, sig, annotations):
        raise NotImplementedError()
    
    def forward_clf(self, segment, segment_avg, inter, inters):
        x = segment
        x_avg = segment_avg
        rr = inters[-self.n_intervals_clf:]
        rr_avg = inters[-self.n_intervals_clf+1:]+[inter]
        x = torch.tensor(x, dtype=torch.float32, device=self.device).view(1, 1, -1)
        x_avg = torch.tensor(x_avg, dtype=torch.float32, device=self.device).view(1, 1, -1)
        rr = torch.tensor(rr, dtype=torch.float32, device=self.device).view(1, -1)
        rr = rr/rr[:, -1]
        rr_avg = torch.tensor(rr_avg, dtype=torch.float32, device=self.device).view(1, -1)
        rr_avg = rr_avg/rr_avg[:, -1]
        out = self.clf(x, x_avg, rr, rr_avg)
        pred = torch.argmax(out).item()
        prob = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
        self.abnormal.append(prob)
        return pred == 0

def convertWindowing(segments, window=(140, 140)):
    converted = []
    l = len(segments[0])
    assert l == 700
    r = len(segments[0])//2
    for segment in segments:
        assert len(segment) == l
        new_segment = segment[r-window[0]:r+window[1]]
        converted.append(new_segment)
    return converted

if __name__ == "__main__":
    # check detector
    from dataset import BaseDataset
    from model import SiameseNetworkRR
    patients = BaseDataset.get_patients('all')
    dataset_path = '../../bnn_ISOCC/mitbih_database/'
    clf = SiameseNetworkRR(input_size=700, n_intervals=10)
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    clf.to(device)
    checkpoint_path='checkpoints/'
    clf.load_state_dict(torch.load(checkpoint_path+'normal_avg.pth'))
    detector = EnhandcedAnomalyDetector(
        clf=clf,
        device=device,
        n_intervals_clf=10,
        std_dist=0.1,
        std_inter=60
    )
    preprocessing = get_preproc('bandpass')
    segmentation = get_segmentation('dynamic_center')
    nn_tn, nn_tp, nn_fn, nn_fp = 0, 0, 0, 0
    for patient in patients:
        print(patient)
        sig, annotations = BaseDataset.load(dataset_path, patient)
        sig = preprocessing((sig-1024)/1024, fc=(1, 150))
        segments, annotations = segmentation(sig, annotations)
        detector.evaluate(segments=segments, annotations=annotations)
        n_tn = len(detector.history['tn'])
        n_tp = len(detector.history['tp'])
        n_fn = len(detector.history['fn'])
        n_fp = len(detector.history['fp'])
        print(len(annotations), '({})'.format(n_tn+n_tp+n_fn+n_fp), n_tn, n_tp, n_fn, n_fp)
        nn_tn += n_tn
        nn_tp += n_tp
        nn_fn += n_fn
        nn_fp += n_fp
    print(*AnomalyDetector.compute_binary_accuracy(nn_tn, nn_tp, nn_fn, nn_fp))
    print('tn: {}  tp: {}  fn: {}  fp: {}'.format(nn_tn, nn_tp, nn_fn, nn_fp))
