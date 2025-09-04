import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import networkx as nx
import pandas as pd
import dgl

from typing import Any, Union, List
from simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def load_anomaly_detection_dataset(dataset, datadir='data'):
    
    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    truth = data_mat['Label']
    truth = truth.flatten()

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    feat = feat.toarray()
    return adj_norm, feat, truth, adj



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./data/processed/{}/{}.mat".format(dataset,dataset))
    text = pd.read_csv('./data/processed/{}/{}.csv'.format(dataset, dataset))


    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))

    return adj, feat, ano_labels, text

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    # nx_graph = nx.from_scipy_sparse_matrix(adj)
    nx_graph = nx.from_scipy_sparse_array(adj)

    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def preprocess_features_ndarray(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(axis=1))  # sum of each row
    r_inv = np.power(rowsum, -1).flatten()  # inverse of row sums
    r_inv[np.isinf(r_inv)] = 0.  # replace inf with 0
    r_mat_inv = np.diag(r_inv)  # create a diagonal matrix with r_inv
    features = r_mat_inv.dot(features)  # row-normalize the feature matrix

    return features

def position_encoding(max_len, emb_size):
    # pe = np.zeros((max_len, emb_size))
    # position = np.arange(0, max_len)[:, np.newaxis]

    pe = np.zeros((max_len, emb_size), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]

    div_term = np.exp(np.arange(0, emb_size, 2) * -(np.log(10000.0) / emb_size))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe



def tokenize(texts: Union[str, List[str]], context_length: int = 128, truncate: bool = True) -> torch.LongTensor:

    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")


        
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

import torch

def compute_cov_eig(X: torch.Tensor, freq=None):
    # X: [N, D]
    X = X.float()
    Xc = X - X.mean(dim=0, keepdim=True)
    # 공분산 (대칭)
    cov = (Xc.T @ Xc) / (X.size(0) - 1)
    # 대칭 행렬 고유분해(grad 가능, 디바이스 그대로)
    lam, U = torch.linalg.eigh(cov)
    return lam, U


def pick_null_dim_by_tail(lam: torch.Tensor, keep_ratio: float = 0.95,    min_null: int = 1,          # d_null 하한
    max_null: int | None = None # 필요하면 상한
):
    """
    lam: 1D tensor of eigenvalues (from torch.linalg.eigh; 보통 오름차순)
    return: (d_row:int, d_null:int)
    """
    assert lam.ndim == 1 and lam.numel() > 0, "lam must be 1D, non-empty"
    # 1) 수치 안정: 음수 클램프, dtype
    lam = lam.to(torch.float64).clamp_min(0)

    # 수치 오차로 인한 아주 작은 음수값 클램프
    lam = torch.clamp(lam, min=0)

    # 2) 내림차순 정렬(큰 것부터 누적)
    lam_sorted, _ = torch.sort(lam, descending=True)

    total = lam_sorted.sum()
    if total <= 0:
        # 전부 0이면 정보가 없으니 전부 null로
        d_row = 0
        d_null = lam_sorted.numel()
    else:
        # 3) keep_ratio 안정화 (1.0 근처 방지)
        keep_ratio = float(min(max(keep_ratio, 0.0), 0.999999))

        var_ratio = lam_sorted / total
        cum = torch.cumsum(var_ratio, dim=0)

        # 4) 처음으로 keep_ratio를 "넘는" 위치 (right=True)
        th = torch.tensor(keep_ratio, dtype=cum.dtype, device=cum.device)
        idx = torch.searchsorted(cum, th, right=True)   # 0-based count

        d_row = int(idx.item())  # 이미 "몇 개 유지할지" 개수
        # 최소 1축은 유지하도록(원하면 제거 가능)
        d_row = max(d_row, 1)
        d_row = min(d_row, lam_sorted.numel())

        d_null = lam_sorted.numel() - d_row

        # 5) null 하한/상한 적용
        if max_null is not None:
            d_null = min(d_null, int(max_null))
        d_null = max(d_null, int(min_null))
        d_row = lam_sorted.numel() - d_null

    return d_row, d_null