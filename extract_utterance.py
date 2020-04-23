#!/usr/bin/env python
# coding: utf-8

import random 
import os
import sys
import csv
import math, pickle
from os.path import isfile, join
import pandas as pd

from data.iemocap import IEMOCAPFeatures

labels = ['hap\t[', 'sad\t[', 'neu\t[', 'ang\t[', 'exc\t[', 'fru\t[']#, 'sur\t[']#'fea\t[', 'dis\t[', 'oth\t[']

def check_label(line):
    for lbl in labels:
        if lbl in line:
            return True
        else:
            continue

trainset = IEMOCAPFeatures()

videoIDs, videoSpeakers, videoLabels, videoText,videoAudio, videoVisual, videoSentence, trainVid,testVid = pickle.load(open('./data/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

#Extract all utterance key
videoList = [v for _, v in videoIDs.items()]
#Unpack the nested list
videokeys = [u for v in videoList for u in v]

#print(len(videokeys))


DATA_PATH = './data/IEMOCAP_full_release/'

sess = ['Session'+str(i) for i in range(1, 6)]

PATHS = sorted([DATA_PATH+p for p in os.listdir(DATA_PATH) if p in sess])
print(PATHS)

#########################################
#  Generate dialoguegcn_utterances.csv
#########################################

#Collect all the utterance file paths
files = []
for p in PATHS:
    files.extend([p+'/dialog/transcriptions/'+s for s in sorted(os.listdir(p+'/dialog/transcriptions/'))])

utterances = []
for f in files:
    with open(f, 'r') as source:
        for line in source:
            if len(line)>0:
                utterances.append(line)

utter_text = [(utter.split()[0], ' '.join(utter.split()[2:])) for utter in utterances]

#Extract line that has utterance
utter_dict = dict()
for rec in utter_text:
    if 'Ses' in rec[0]:
        utter_dict[rec[0]] = rec[1]
        
df = pd.DataFrame.from_dict(utter_dict, orient='index', columns=['text'])

df = df[df.index.isin(videokeys)]

df.to_csv('./data/dialoguegcn_utterances.csv')


# In[60]:


#########################################
#  Generate emotion_labels.csv
#########################################

#Collect all the emotion label file paths
files = []
for p in PATHS:
    files.extend([p+'/dialog/EmoEvaluation/'+s for s in sorted(os.listdir(p+'/dialog/EmoEvaluation/')) if s[-3:]=='txt'])

corpus = []
for f in files:
    with open(f, 'r') as source:
        for line in source:
            if len(line)>0:
                corpus.append(line)
                
#Extract line that contains the 6 emotion labels
annotations = [line for line in corpus if check_label(line)]

#Create dict to store utterance key and label
annot_list = [annot.split('\t')[1:3] for annot in annotations]
emotion_labels = dict()
for rec in annot_list:
    emotion_labels[rec[0]] = rec[1]

emo = pd.DataFrame.from_dict(emotion_labels, orient='index', columns=['label'])
#print(emo.shape)

df = df.join(emo)
df = df.dropna()
#print(df.shape)

df.to_csv('./data/emotion_labels.csv')




