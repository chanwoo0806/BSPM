import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
from torch import spmm

def get_norm_inter(inter):
    user_degree = np.array(inter.sum(axis=1)).flatten() # Du
    item_degree = np.array(inter.sum(axis=0)).flatten() # Di
    user_d_inv_sqrt = np.power(user_degree.clip(min=1), -0.5)
    item_d_inv_sqrt = np.power(item_degree.clip(min=1), -0.5)
    user_d_inv_sqrt[user_degree == 0] = 0
    item_d_inv_sqrt[item_degree == 0] = 0
    user_d_inv_sqrt = sp.diags(user_d_inv_sqrt)  # Du^(-0.5)
    item_d_inv_sqrt = sp.diags(item_d_inv_sqrt)  # Di^(-0.5)
    norm_inter = (user_d_inv_sqrt @ inter @ item_d_inv_sqrt).tocoo() # Du^(-0.5) * R * Di^(-0.5)
    return norm_inter # R_tilde

def sparse_coo_tensor(mat):
    # scipy.sparse.coo_matrix -> torch.sparse.coo_tensor
    return torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack([mat.row, mat.col])),
        values=torch.tensor(mat.data, dtype=torch.float32),
        size=mat.shape)

class IdealFilter(nn.Module):
    def __init__(self, threshold, weight):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
    
    def fit(self, inter):
        norm_inter = get_norm_inter(inter)
        _, _, vt = svds(norm_inter, which='LM', k=self.threshold)
        ideal_pass = torch.tensor(vt.T.copy())
        self.register_buffer('ideal_pass', ideal_pass) # shape (num_items, threshold)
        
    def forward(self, signal):
        ideal_preds = signal @ self.ideal_pass @ self.ideal_pass.T
        return ideal_preds * self.weight

class DegreeNorm(nn.Module):
    def __init__(self, power):
        super().__init__()
        self.power = power
    
    def fit(self, inter):
        item_degree = torch.tensor(np.array(inter.sum(axis=0)).flatten())
        zero_mask = (item_degree == 0)
        pre_norm = item_degree.clamp(min=1).pow(-self.power)
        pst_norm = item_degree.clamp(min=1).pow(+self.power)
        pre_norm[zero_mask], pst_norm[zero_mask] = 0, 0
        self.register_buffer('pre_normalize', pre_norm)  # (num_items,)
        self.register_buffer('post_normalize', pst_norm) # (num_items,)
        
    def forward_pre(self, signal):
        return signal * self.pre_normalize
    
    def forward_post(self, signal):
        return signal * self.post_normalize

class LinearFilter(nn.Module):
    def __init__(self):
        super().__init__()
        
    def fit(self, inter):
        norm_inter = get_norm_inter(inter)
        norm_inter = sparse_coo_tensor(norm_inter)
        self.register_buffer('norm_inter', norm_inter) # shape (num_users, num_items)
        
    def forward(self, signal):
        signal = signal.T
        output = spmm(self.norm_inter.t(), spmm(self.norm_inter, signal))
        return output.T
    
    def sharpen(self, signal):
        return self.forward(signal) * (-1)