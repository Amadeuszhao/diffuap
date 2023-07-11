import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss

def one_hot(class_labels, num_classes=None):
    if num_classes==None:
        return torch.zeros(len(class_labels), class_labels.max()+1).scatter_(1, class_labels.unsqueeze(1), 1.)
    else:
        return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.)
    

class BoundedLogitLossFixedRef(_WeightedLoss):
    def __init__(self, num_classes=1000, confidence=0.0, use_cuda=False):
        super(BoundedLogitLossFixedRef, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = use_cuda

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)
        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(not_target_logits.data.detach() - target_logits, min=-self.confidence)
        return torch.mean(logit_loss)

if __name__ == '__main__':
    loss = BoundedLogitLossFixedRef()
    a = torch.rand(1,1000)
    b = torch.tensor([483])
    print(a.shape)
    print(b.shape)
    print(loss(a,b))