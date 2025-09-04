import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution
from utils import compute_cov_eig, pick_null_dim_by_tail

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x

class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()
        
        self.feat_size = feat_size
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)


          # X_row_init: (N, d_row), torch.float32, 사전 계산된 whitening 결과
        self.E_lang = nn.Embedding.from_pretrained(X_row_init, freeze=True)
        self.E_null = nn.Embedding(num_nodes, d_null)
        nn.init.normal_(self.E_null.weight, mean=0.0, std=0.02)

        self.proj = nn.Linear(feat_size, d_null, bias=False)  # 구조 임베딩 투사
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, x, adj, text_emb):

        eps = 1e-12
        lam, U = compute_cov_eig(text_emb, freq=None)  # svd

        d_row, d_null = pick_null_dim_by_tail(lam, keep_ratio=0.95) #null space 뽑음

        U_row = U[:, :d_row]                    # d_row 개만
        S_row_inv_sqrt = torch.diag(1.0 / torch.sqrt(lam[:d_row] + eps)) # whitening singular value
        P_row = U_row @ S_row_inv_sqrt     # 그런 space

        X = text_emb.to(torch.float64)
        mu = X.mean(axis=0, keepdims=True)
        X_row = (X - mu) @ P_row        # space 위로 투영

        E_lang = torch.nn.Embedding.from_pretrained(
            torch.tensor(X_row, dtype=torch.float32), freeze=True
        ) # 임베딩 곶어

        E_null = torch.nn.Embedding(num_embeddings=X.shape[0], embedding_dim=d_null) # null slot
        torch.nn.init.normal_(E_null.weight, mean=0.0, std=0.02)


        # X_row: (N, d_row) — numpy or torch
        X_row_t = torch.tensor(X_row, dtype=torch.float32,)  # freeze 용
        pad     = torch.zeros(X_row_t.size(0), d_null, dtype=X_row_t.dtype)
        Z_full  = torch.cat([X_row_t, pad], dim=-1)  # (N, d_row + d_null)


        # H_struct: (N, d_struct)
        proj = torch.nn.Linear(self.feat_size, d_null, bias=False)
        torch.nn.init.xavier_uniform_(proj.weight)
        H_null = proj(x)                  # (N, d_null)
        # (정규화가 필요하면) H_null = torch.nn.functional.normalize(H_null, dim=-1)
        Z_full[..., -d_null:] += H_null   # 옵션

      
        # encode
        x = self.shared_encoder(Z_full, adj)
        # decode feature matrix
        x_hat = self.attr_decoder(x, adj)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(x, adj)
        # return reconstructed matrices


        return struct_reconstructed, x_hat
        