import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def loss_function(pred, label, device):
    
    # weights for unbalanced classes
    loss_weights = torch.FloatTensor([1/0.086747,
                                      1/0.144406,
                                      1/0.227883,
                                      1/0.160585,
                                      1/0.127711,
                                      1/0.252668]).to(device)

    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.NLLLoss(loss_weights)
    logsoftmax = nn.LogSoftmax(dim=1)
    loss = criterion(logsoftmax(pred), label)
    return loss

# def loss(pred, label, device):

#     # calculating label weights for weighted loss computation
#     V = label.size(0)
#     label_count = torch.bincount(label)
#     label_count = label_count[label_count.nonzero()].squeeze()
#     cluster_sizes = torch.zeros(self.n_classes).long().to(device)
#     cluster_sizes[torch.unique(label)] = label_count
#     weight = (V - cluster_sizes).float() / V
#     weight *= (cluster_sizes>0).float()

#     # weighted cross-entropy for unbalanced classes
#     criterion = nn.CrossEntropyLoss(weight=weight)
#     loss = criterion(pred, label)

#     return loss