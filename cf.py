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
        self.user_emb.weight.data.uniform_(0, 0.5)
        self.item_emb.weight.data.uniform_(0, 0.5)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        return (u * i).sum(1) 
        # instead of dot product, use some network and return softmaxed output to use nll_loss
