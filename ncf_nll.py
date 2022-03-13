from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class CollaborativeFiltering(Module):
    def __init__(self, n_users, n_items, n_factors):
        super(CollaborativeFiltering, self).__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.fc = nn.Linear(n_factors * 2, 1)
        self.user_emb.weight.data.uniform_(0, 0.5)
        self.item_emb.weight.data.uniform_(0, 0.5)
        self.fc.weight.data.uniform_(0, 0.5)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        features = torch.concat([u, i], dim = 1)
        x = self.fc(features)
        out = torch.sigmoid(x)
        return out

# Reference code: https://www.kaggle.com/shahrukhkhan/rec-sys-neural-collaborative-filtering-pytorch

# Cross-entropy loss for this CF architecture can be used like this -
# For all pairs of (u, i) in train-edges, add the following to "loss"
# loss += - A(u, i) * log (y_hat(u, i)) - (1 - A(u, i)) log (1 - y_hat(u, i))