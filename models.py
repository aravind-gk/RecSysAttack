import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class CF(Module):
    """Simple Collaborative Filtering"""
    def __init__(self, n_users, n_items, n_factors, prob = None):
        super(CF, self).__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        dot = (u * i).sum(1)
        return torch.sigmoid(dot)

class CFD(Module):
    """Simple Collaborative Filtering with dropout"""
    def __init__(self, n_users, n_items, n_factors, prob = 0.5):
        super(CFD, self).__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.drop_u = nn.Dropout(p = prob)
        self.drop_i = nn.Dropout(p = prob)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        u = self.drop_u(u)
        i = self.drop_i(i)
        dot = (u * i).sum(1)
        return torch.sigmoid(dot)

class GMF(Module):
    """General Matrix Factorization with 1 hidden layer"""
    def __init__(self, n_users, n_items, n_factors, prob = None):
        super(GMF, self).__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.h = nn.Linear(n_factors, 1)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        dot = (u * i)
        x = self.h(dot).squeeze()
        return torch.sigmoid(x)

class GMFD(Module):
    """General Matrix Factorization with 1 hidden layer and dropout"""
    def __init__(self, n_users, n_items, n_factors, prob = 0.5):
        super(GMFD, self).__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.h = nn.Linear(n_factors, 1)
        self.drop_u = nn.Dropout(p = prob)
        self.drop_i = nn.Dropout(p = prob)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        u = self.drop_u(u)
        i = self.drop_i(i)
        dot = (u * i)
        x = self.h(dot).squeeze()
        return torch.sigmoid(x)

# GMF with user-item bias and dropout
class GMFB(Module):
    """General Matrix Factorization with user-item bias, 1 hidden layer and dropout"""
    def __init__(self, n_users, n_items, n_factors, prob = 0.5):
        super(GMFB, self).__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.h = nn.Linear(n_factors * 3, 1)
        self.drop_u = nn.Dropout(p = prob)
        self.drop_i = nn.Dropout(p = prob)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        u = self.drop_u(u)
        i = self.drop_i(i)
        dot = (u * i)
        feat = torch.concat([dot, u, i], 1)
        x = self.h(feat).squeeze()
        return torch.sigmoid(x)

class MLP(Module):
    """Multi-layer Perceptron without hadamard product"""
    def __init__(self, n_users, n_items, n_factors, prob = 0.5):
        super(MLP, self).__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.h = nn.Linear(n_factors * 2, n_factors * 2)
        self.o = nn.Linear(n_factors * 2, 1)
        self.drop_u = nn.Dropout(p = prob)
        self.drop_i = nn.Dropout(p = prob)
        self.drop_x = nn.Dropout(p = prob)
        self.tanh = nn.Tanh()

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        u = self.drop_u(u)
        i = self.drop_i(i)
        x = torch.concat([u, i], 1)
        # x = self.tanh(x)
        x = self.h(x)
        # x = self.drop_x(x)
        # x = F.relu(x) # 1 (both relu and sigmoid seem to work fine here)
        x = torch.sigmoid(x) # 2
        # x = self.tanh(x) # 3
        x = self.o(x)
        return torch.sigmoid(x.squeeze())

class NeuMF(Module):
    """NeuMF combining 1-layer GMF and 2-layer MLP with additional layer"""
    def __init__(self, n_users, n_items, n_factors, prob = 0.5):
        super(NeuMF, self).__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.gmf = nn.Linear(n_factors, n_factors)
        self.mlp1 = nn.Linear(n_factors * 2, n_factors * 2)
        self.mlp2 = nn.Linear(n_factors * 2, n_factors)
        self.out = nn.Linear(n_factors * 2, 1)
        
        self.drop = nn.Dropout(p = prob)
        self.tanh = nn.Tanh()

    def forward(self, user, item):
        user = self.user_emb(user)
        item = self.item_emb(item)
        user = self.drop(user)
        item = self.drop(item)

        gmf = (user * item)
        gmf = self.gmf(gmf)

        mlp = torch.concat([user, item], 1)
        mlp = self.mlp1(mlp)
        mlp = torch.sigmoid(mlp)
        mlp = self.mlp2(mlp)

        neumf = torch.concat([gmf, mlp], 1)
        neumf = self.drop(neumf)
        neumf = self.out(neumf)
        neumf = neumf.squeeze()

        return torch.sigmoid(neumf)

def get_accuracy(y_hat, y):
    y = y.clone().int()
    y_hat = (y_hat.clone() > 0.5).int()
    accuracy = (y == y_hat).sum() / len(y)
    return accuracy.item()