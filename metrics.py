import numpy as np
import torch

class BaseMetric():
    def __init__(self, target=None):
        self.value = 0
        self.target = target

    def update(self, preds, labels, loss):
        raise NotImplementedError()

    def reset(self):
        self.value = 0

class AvgLoss(BaseMetric):
    def __init__(self, target=None):
        super().__init__(target=target)
        self.count = 0
    
    def update(self, preds, labels, loss):
        loss = loss.item()
        self.value = (self.value*self.count+loss)/(self.count+1)
        self.count += 1
    
    def reset(self):
        super().reset()
        self.count = 0

class Accuracy(BaseMetric):
    def __init__(self, target=None):
        super().__init__(target=target)
        self.n_correct = 0
        self.n_total = 0
    
    def update(self, preds, labels, loss):
        if self.target is None:
            # compute average accuracy for all classes

            n_correct = (preds == labels).sum().item()
            n_total = preds.shape[0]
        else:
            # accuracy for the target class
            n_correct = torch.logical_and(preds == self.target, preds == labels).sum().item()
            n_total = (labels == self.target).sum().item()
        self.n_correct += n_correct
        self.n_total += n_total
        self.value = 0 if self.n_total == 0 else self.n_correct/self.n_total
    
    def reset(self):
        super().reset()
        self.n_correct = 0
        self.n_total = 0
    
class Sensitivity(BaseMetric):
    def __init__(self, target=None):
        super().__init__(target=target)
        if self.target is None:
            raise ValueError('specify target class')
        self.n_tp = 0
        self.n_target = 0
    
    def update(self, preds, labels, loss):
        # Accuracy랑 같음

        n_tp = torch.logical_and(preds == self.target, labels == self.target).sum().item()
        n_target = (labels == self.target).sum().item()
        self.n_tp += n_tp
        self.n_target += n_target
        self.value = 0 if self.n_target == 0 else self.n_tp/self.n_target
    
    def reset(self):
        super().reset()
        self.n_tp = 0
        self.n_target = 0

class Specificity(BaseMetric):
    def __init__(self, target=None):
        super().__init__(target=target)
        if self.target is None:
            raise ValueError('specify target class')
        self.n_tp = 0
        self.n_pred = 0
    
    def update(self, preds, labels, loss):

        n_tp = torch.logical_and(preds == self.target, labels == self.target).sum().item()
        n_pred = (preds == self.target).sum().item()
        self.n_tp += n_tp
        self.n_pred += n_pred
        self.value = 0 if self.n_pred == 0 else self.n_tp/self.n_pred
    
    def reset(self):
        super().reset()
        self.n_tp = 0
        self.n_pred = 0

class ConfusionMatrix(BaseMetric):
    def __init__(self, target=None):
        super().__init__(target=target)
        pass

class F1score(BaseMetric):
    def __init__(self, target=None):
        super().__init__(target=target)
        pass

def print_metrics(metrics, str=""):
    for metric in metrics:
        str += '{}: {:.3f}  '.format(metric, metrics[metric].value)
    print(str)

def reset_metrics(metrics):
    for metric in metrics:
        metrics[metric].reset()
