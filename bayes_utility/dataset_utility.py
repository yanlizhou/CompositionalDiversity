# import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

from .render import primitives, render_from_string


def get_probs(neg_set):
    l3s = sum(list(map(lambda x: len(x.split('+')[0])==5, neg_set)))
    l2s = sum(list(map(lambda x: len(x.split('+')[0])==3, neg_set)))
    l1s = sum(list(map(lambda x: len(x.split('+')[0])==1, neg_set)))

    p1 = (1/3)/l1s
    p2 = (1/3)/l2s
    p3 = (1/3)/l3s
    
    probs = list(map(lambda x : p1 if len(x.split('+')[0])==1 else (p2 if len(x.split('+')[0])==3 else p3),
                     neg_set))
    return probs
        
class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class dataset(Dataset):
    def __init__(self, dataset_type, n_support = 8, 
        n_choice = 8, n_pos=4, n_neg=4, root_dir='./data/', transform=None, if_resnet = False):

        self.root_dir = root_dir
        self.transform = transform
        self.file_name = root_dir+dataset_type+'.pkl'
        self.n_support = n_support
        self.n_choice = n_choice
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.if_resnet = if_resnet

        with open(self.file_name, 'rb') as f:
            self.hypotheses = pickle.load(f)
        
        with open('./data/all_tokens.pkl', 'rb') as f:
            self.all_tokens = pickle.load(f)

    def __len__(self):
        return len(self.hypotheses)

    def __getitem__(self, idx):
        prims = np.random.choice(np.arange(len(primitives)),4,replace=False)

        h = self.hypotheses[idx]
        h_set = h()

        question_set = np.random.choice(list(h_set),self.n_support)

        positive = np.random.choice(list(h_set), self.n_pos)
        negative_set = list(self.all_tokens.difference(h_set))
        negatives = np.random.choice(negative_set, self.n_neg, p=get_probs(negative_set))
        answer_set = np.concatenate((positive,negatives))

        images = []
        targets = np.zeros(self.n_choice)
        my_perm = np.random.permutation(self.n_choice)
        for s in question_set:
            images.append(np.array(render_from_string(s, prims, if_resnet=self.if_resnet)))
        for i,ind in enumerate(my_perm):
            images.append(np.array(render_from_string(answer_set[ind], prims, if_resnet=self.if_resnet)))
            if ind<self.n_pos:
                targets[i] = 1

        images = np.stack(images)

        if self.transform:
            images = self.transform(images)
        
        targets = torch.tensor(targets, dtype=torch.float32)



        return images, targets
