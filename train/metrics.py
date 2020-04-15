import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support

def accuracy_emotion_like_SBM_weighted(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=0)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes)/ float(nb_non_empty_classes)
    return acc

def f1_emotion_like_dialogueGCN_paper(scores, targets):
    # scores = scores.detach().argmax(dim=1).cpu()
    scores = np.argmax( torch.nn.Softmax(dim=0)(scores).cpu().detach().numpy() , axis=1 )
    targets = targets.cpu().detach().numpy()
    return round(f1_score(targets, scores, average='weighted')*100, 2)

def accuracy_emotion_like_dialogueGCN_paper(scores, targets):
    # scores = scores.detach().argmax(dim=1).cpu()
    scores = np.argmax( torch.nn.Softmax(dim=0)(scores).cpu().detach().numpy() , axis=1 )
    targets = targets.cpu().detach().numpy()
    return round(accuracy_score(targets, scores)*100, 2)
    
def accuracy_VOC(scores, targets):
    scores = scores.detach().argmax(dim=1).cpu()
    targets = targets.cpu().detach().numpy()
    acc = f1_score(scores, targets, average='weighted')
    return acc
