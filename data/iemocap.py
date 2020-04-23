import numpy as np, argparse, time, pickle, random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import random

import os
import pickle, pandas as pd
from torch.nn.utils.rnn import pad_sequence
from scipy.spatial.distance import cdist
from scipy import ndimage
import numpy as np

import dgl

"""
    # Output of the trainset or testset contains a Tensor of num_samples data.

    Each sample consists:
    {N_Utter is the number of utterances in one data sample}

    0 -> text features; dim: N_Utter x 100
    1 -> visual features; dim: N_Utter x 512
    2 -> audio features; dim: N_utter x 100
    3 -> speaker gender ([1,0] for Male; [0,1] for Female)
    4 -> ones Tensor of size N_Utter
    5 -> labels of each utterance; label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
    6 -> video key; eg. Ses05F_script01_2

"""



class IEMOCAPFeatures(Dataset):

    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('data/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]



"""
    To change kth, make changes in the arguments in line 73, 81 and 108.
"""
    
    
def sigma(dists, kth=25):
    """
    # Get k-nearest neighbors for each node
    knns = np.partition(dists, kth, axis=-1)[:, kth::-1]

    # Compute sigma and reshape
    sigma = knns.sum(axis=1).reshape((knns.shape[0], 1))/kth
    return sigma + 1e-8 # adding epsilon to avoid zero value of sigma
    """

    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1))/kth
    except ValueError:     # handling for graphs with num_nodes less than kth
        num_nodes = dists.shape[0]
        # this sigma value is irrelevant since not used for final compute_edge_list
        sigma = np.array([1]*num_nodes).reshape(num_nodes,1)
        
    return sigma + 1e-8 # adding epsilon to avoid zero value of sigma

def compute_adjacency_matrix(audio, visual, use_audio=True, use_visual=True, kth=25):
    # audio = audio.reshape(-1, 2)
    
    if use_audio and use_visual:
        audio_dist = cdist(audio, audio)     # Compute audio based distance
        visual_dist = cdist(visual, visual)  # Compute visual based distance

        # Compute adjacency
        A = np.exp(- (audio_dist/sigma(audio_dist, kth))**2 - (visual_dist/sigma(visual_dist, kth))**2 )
        
    elif use_audio:
        audio_dist = cdist(audio, audio)     # Compute audio based distance
        
        # Compute adjacency
        A = np.exp(- (audio_dist/sigma(audio_dist, kth))**2)
        
    elif use_visual:
        visual_dist = cdist(visual, visual)  # Compute visual based distance
        
        # Compute adjacency
        A = np.exp(- (visual_dist/sigma(visual_dist, kth))**2)
        
    # Convert adjacency to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A

def compute_edges_list(A, kth):
    """
    # Get k-similar neighbor indices for each node
    if 1==1:   
        num_nodes = A.shape[0]
        new_kth = num_nodes - kth
        knns = np.argpartition(A, new_kth-1, axis=-1)[:, new_kth:-1]
        knns_d = np.partition(A, new_kth-1, axis=-1)[:, new_kth:-1]
    else:
        knns = np.argpartition(A, kth, axis=-1)[:, kth::-1]
        knns_d = np.partition(A, kth, axis=-1)[:, kth::-1]
    return knns, knns_d
    """
    # Get k-similar neighbor indices for each node

    num_nodes = A.shape[0]
    new_kth = num_nodes - kth
    
    if num_nodes > kth:
        knns = np.argpartition(A, new_kth-1, axis=-1)[:, new_kth:-1]
        knn_d = np.partition(A, new_kth-1, axis=-1)[:, new_kth:-1]
    else:
        # handling for graphs with less than kth nodes
        # in such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_d = A # NEW
        
        # removing self loop
        knn_d = A[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1) 
        knns = knns[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1)
    return knns, knn_d 



class IEMOCAP_KNN_Graph(torch.utils.data.Dataset):
    def __init__(self, split, use_audio=True, use_visual=True):
        self.split = split
        self.graph_lists = []
        
        if split == 'train':
            self.dataset = IEMOCAPFeatures()
        else:
            self.dataset = IEMOCAPFeatures(train=False)
        
        self.kth = 25
        self.use_audio = use_audio
        self.use_visual = use_visual
        
        self.n_samples = len(self.dataset)
        self._prepare()
        
    def _prepare(self):
        t0 = time.time()
        print("[I] Preparing IEMOCAPDataset as a {}-NN similarity graph for split {} ...".format(self.kth, self.split))
        self.adj_matrices, self.node_features, self.edge_features, self.edges_lists, self.labels = [], [], [], [], []
        for sample in self.dataset:
            visual, audio = sample[1], sample[2]
            A = compute_adjacency_matrix(audio, visual, self.use_audio, self.use_visual, self.kth)
            edges_list, edge_values_list = compute_edges_list(A, self.kth+1)
            edge_values_list = edge_values_list.reshape(-1)

            self.node_features.append(sample[0])      # using text features as node features
            self.edge_features.append(edge_values_list)
            self.adj_matrices.append(A)
            self.edges_lists.append(edges_list)
            self.labels.append(sample[5])
            
            
        for index in range(len(self.dataset)):
            g = dgl.DGLGraph()
            g.add_nodes(self.node_features[index].shape[0])
            g.ndata['feat'] = torch.Tensor(self.node_features[index])
            for src, dsts in enumerate(self.edges_lists[index]):
                g.add_edges(src, dsts[dsts!=src])
                
            # adding edge features for GatedGCN
            # edge_feat_dim = g.ndata['feat'].shape[1] # dim same as node feature dim
            # g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)
            g.edata['feat'] = torch.Tensor(self.edge_features[index]).unsqueeze(1)

            self.graph_lists.append(g)
        print("[I] Finished preparation after {:.4f}s".format(time.time()-t0))
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.graph_lists[idx], self.labels[idx]
    
    
class IEMOCAP_KNN_Graph_Dataset(torch.utils.data.Dataset):
    def __init__(self, name, num_val=30):
        t_data = time.time()
        self.name = name

        use_audio = True
        use_visual = True
        
        print("[I] For Graph Construction: \n[I] Using audio: {}\n[I] Using video: {}\n".format(use_audio, use_visual))
        
        self.test = IEMOCAP_KNN_Graph(split='test', use_audio=use_audio, use_visual=use_visual)

        self.train = IEMOCAP_KNN_Graph(split='train', use_audio=use_audio, use_visual=use_visual)

        print("[I] Data load time: {:.4f}s".format(time.time()-t_data))
        
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n, snorm_e