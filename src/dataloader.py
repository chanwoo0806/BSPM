import logging
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

def get_interaction(dataset):
    '''
    train_inter: scipy.sparse.coo_matrix of shape(num_users, num_items)
    test_inter: dict of (key = user id, value = a set of item ids)
    '''
    train_path = f'./dataset/{dataset}/train.txt'
    test_path = f'./dataset/{dataset}/test.txt'
    max_uid, max_iid = -1, -1
    
    with open(train_path, "r") as f:
        lines = f.readlines()
    rows, cols = list(), list()
    num_train = 0
    for line in lines:
        line = [int(idx) for idx in line.strip().split(" ")]
        uid, iids = line[0], line[1:] # first index is user id, the rest are item ids
        iids = list(set(iids)) # remove duplicates
        rows.extend([uid] * len(iids))
        cols.extend(iids)
        num_train += len(iids)
        max_uid = max(max_uid, uid)
        max_iid = max(max_iid, *iids)

    with open(test_path, "r") as f:
        lines = f.readlines()
    test_inter = dict()
    num_test = 0
    for line in lines:
        line = [int(idx) for idx in line.strip().split(" ")]
        uid, iids = line[0], line[1:] # first index is user id, the rest are item ids
        if len(iids) == 0: continue # skip users without interaction
        iids = set(iids) # remove duplicates
        test_inter[uid] = iids
        num_test += len(iids)
        max_uid = max(max_uid, uid)
        max_iid = max(max_iid, *iids)
        
    rows, cols = np.array(rows), np.array(cols)
    vals = np.ones(len(rows), dtype=np.float32)
    train_inter = sp.coo_matrix((vals, (rows, cols)), shape=(max_uid+1, max_iid+1))
    return train_inter, num_train, test_inter, num_test

class AllRankData(Dataset):
    def __init__(self, observed_inter, test_inter):
        self.observed_inter = observed_inter.tocsr()
        self.test_inter = test_inter
        self.test_users = list(test_inter.keys())
        
    def __len__(self):
        return len(self.test_users)
    
    def __getitem__(self, idx):
        uid = self.test_users[idx]
        observed = self.observed_inter[uid].toarray().flatten()
        label = self.test_inter[uid]
        return observed, label
    
def collate_fn(batch, device):
    observed_inters, label_inters = zip(*batch)
    observed_inters = torch.tensor(np.stack(observed_inters)).to(device)
    return observed_inters, label_inters

def load_data(dataset, batch_size, device):
    train_inter, num_train, test_inter, num_test = get_interaction(dataset)
    test_data = AllRankData(train_inter, test_inter)
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=lambda batch: collate_fn(batch, device))
    # Log dataset statistics
    logging.info(f'[DATASET] {dataset}')
    logging.info(f'# of Users: {train_inter.shape[0]}')
    logging.info(f'# of Items: {train_inter.shape[1]}')
    logging.info(f'# of Training Interactions: {num_train}')
    logging.info(f'# of Testing Interactions: {num_test}')
    logging.info('')
    return train_inter, test_dataloader