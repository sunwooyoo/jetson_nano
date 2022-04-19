import statistics
import pywt
import numpy as np
from scipy.signal import firwin, filtfilt

normal_beats = ['N', 'L', 'R', 'e', 'j']
sup_beats = ['A', 'a', 'J', 'S']
ven_beats = ['V', 'E']
fusion_beats = ['F']
unknown_beat = ['/', 'f', 'Q']

def wavelet_denoising(x, **kwargs):
    '''
    arguments:
        `x` (array-like): get_preproc
        `wavelet` (str): wavelet function
        `max_level` (int): max level for WT
    '''
    w = 'db6' if 'wavelet' not in kwargs else kwargs['wavelet']
    max_level = 9 if 'max_level' not in kwargs else kwargs['max_level']

    w = pywt.Wavelet(w)
    coeffs = pywt.wavedec(x, w, level=max_level)

    for i in range(1, len(coeffs)):
        threshold = np.sqrt(2*np.log(len(x)))*statistics.median(abs(coeffs[1]))/0.6745
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]), 'hard')
    
    coeffs[0] = np.zeros(coeffs[0].shape)
    coeffs[-1] = np.zeros(coeffs[-1].shape)
    coeffs[-2] = np.zeros(coeffs[-2].shape)

    return pywt.waverec(coeffs, w)
def fir_bandpass(sig, **kwargs):
    num_taps = 2000 if 'num_taps' not in kwargs else kwargs['num_taps']
    fc = (1.5, 150) if 'fc' not in kwargs else kwargs['fc']
    fs = 360 if 'fs' not in kwargs else kwargs['fs']
    fc1, fc2 = fc
    nyq = 0.5*fs
    h = firwin(num_taps, [fc1/nyq, fc2/nyq], pass_zero=False)

    return filtfilt(h, [1.0], sig).copy()
    
def no_preproc(sig, **kwargs):
    return sig

def get_preproc(name):
    PREPROC = {
        'wavelet': wavelet_denoising,
        'bandpass': fir_bandpass,
        'none': no_preproc
    }
    if name not in PREPROC:
        print('function for '+name+' is not implemented')
        print('available options: {}'.format([p for p in PREPROC]))
        raise Exception
    else:
        return PREPROC[name]


#### segmentation
def windowing(x, annotations, **kwargs):
    signal_len = len(x)
    window = (140, 139) if 'window' not in kwargs else kwargs['window']
    window_left, window_right = window
    segments = []
    new_annotations = []
    for annotation in annotations:
        r_position = int(annotation['Sample#'])
        if r_position < window_left:
            continue
        elif r_position > signal_len - window_right:
            continue
        else:
            segment = x[r_position-window_left:r_position+window_right]
            segments.append(segment)
            new_annotations.append(annotation)
    return segments, new_annotations

def dynamic_windowing_center(x, annotations, **kwargs):
    min_size = 100 if 'min_size' not in kwargs else kwargs['min_size']
    size = 700 if 'size' not in kwargs else kwargs['size']
    segments = []
    new_annotations = []
    for i in range(1, len(annotations)-1):
        r_cur = int(annotations[i]['Sample#'])
        r_pre = int(annotations[i-1]['Sample#']) + 50
        r_next = int(annotations[i+1]['Sample#']) - 50
        if r_next - r_pre < min_size:
            continue
        if r_pre > r_cur:
            continue
        if r_next < r_cur:
            continue
        if r_cur - r_pre > size//2:
            segment_left = x[r_cur-size//2:r_cur]
        else:
            len_padding = size//2-(r_cur-r_pre)
            padding_value = x[r_pre]
            segment_left = np.concatenate([np.ones(len_padding)*padding_value, x[r_pre:r_cur]])
            assert len(segment_left) == size//2
        if r_next - r_cur > size//2:
            segment_right = x[r_cur:r_cur+size//2]
        else:
            len_padding = size//2-(r_next-r_cur)
            padding_value = x[r_next-1]
            segment_right = np.concatenate([x[r_cur:r_next], np.ones(len_padding)*padding_value])
            assert len(segment_right) == size//2
        segment = np.concatenate([segment_left, segment_right])
        segments.append(segment)
        new_annotations.append(annotations[i])
    return segments, new_annotations


def dynamic_windowing_resamp(x, annotations, **kwargs):
    pass
        


def get_segmentation(name):
    SEGMENTATION = {
        'windowing': windowing,
        'dynamic_center': dynamic_windowing_center
    }
    if name not in SEGMENTATION:
        print('function for '+name+' is not implemented')
        print('available options: {}'.format([s for s in SEGMENTATION]))
        raise Exception
    else:
        return SEGMENTATION[name]
