from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from datetime import datetime
import argparse

from model_text import Dominant
from utils import load_anomaly_detection_dataset, adj_to_dgl_graph, preprocess_features_ndarray, load_mat
from sklearn import preprocessing
from data import DataHelper
from torch.utils.data import DataLoader
import scipy.sparse as sp

from utils import tokenize
from sentence_transformers import SentenceTransformer



def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)


    cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def train_dominant(args):

    # adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)

    # adj = torch.FloatTensor(adj)
    # adj_label = torch.FloatTensor(adj_label)
    # attrs = torch.FloatTensor(attrs)

   # ---- 1) 데이터 로드: 기존 load_mat 그대로 사용 ----
    # adj(=잘못된 D일 수 있음), attrs(scipy), label(np), text(pd.DataFrame)
    adj_bad, attrs_sp, label, text = load_mat(args.dataset)

    # !!! dgl_graph/edges()에서 뽑은 arr_edge_index가 유효하다는 전제 하에 사용
    dgl_graph = adj_to_dgl_graph(adj_bad)
    src = dgl_graph.edges()[0].numpy()
    dst = dgl_graph.edges()[1].numpy()
    arr_edge_index = np.vstack((src, dst))

    # ---- 2) arr_edge_index → A (대칭화 + 대각 0) ----
    src, dst = arr_edge_index
    num_nodes = int(max(src.max(), dst.max()) + 1)

    A = sp.csr_matrix(
        (np.ones_like(src, dtype=np.float32), (src, dst)),
        shape=(num_nodes, num_nodes)
    )
    A = A.maximum(A.T)          # 무방향이면 대칭화
    A = A.tolil()
    A.setdiag(0.0)              # ★ 대각 0으로 리셋
    A = A.tocsr()

    A_label = A + sp.eye(num_nodes, dtype=np.float32, format='csr')  # ★ 타깃은 A+I

    # ---- 3) scipy.sparse → torch.sparse ----
    def sp_to_torch_sparse(m):
        coo = m.tocoo()
        idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        val = torch.tensor(coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, torch.Size(coo.shape))

    adj = sp_to_torch_sparse(A).coalesce().to(args.device)
    adj_label = sp_to_torch_sparse(A_label).coalesce().to(args.device)

    # ---- 4) 특징행렬 X: 정규화된 dense 텐서 ----
    # attrs_sp: (N,F) scipy.sparse
    X_np = attrs_sp.toarray()
    X_np = preprocessing.StandardScaler().fit_transform(X_np)
    X = torch.tensor(X_np, dtype=torch.float32, device=args.device)

    # ---- 5) 모델 준비 ----
    model = Dominant(
        feat_size=X.size(1),
        hidden_size=args.hidden_dim,
        dropout=args.dropout
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Data = DataHelper(arr_edge_index, args)
    # loader = DataLoader(Data, batch_size=args.batch_size, shuffle=True, num_workers=10)

    tit_list = text['text'].to_numpy()
    text_token = tokenize(tit_list, context_length=args.context_length)
    st = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)
    emb_text = st.encode(tit_list.tolist(),
                             normalize_embeddings=True, device=args.device) 
    emb_text = torch.tensor(emb_text, dtype=torch.float32, device=args.device) # (N, 384)

    for epoch in range(args.epoch):

        optimizer.zero_grad()

        A_hat, X_hat = model(X, adj, emb_text)
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, X, X_hat, args.alpha)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()        
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

        if epoch%10 == 0 or epoch == args.epoch - 1:
            model.eval()
            A_hat, X_hat = model(X, adj , emb_text)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, X, X_hat, args.alpha)
            score = loss.detach().cpu().numpy()
            # print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))
                    
            print("Epoch:", '%04d' % (epoch),
                'Auc', roc_auc_score(label, score),
                'Precision', precision_score(label, score > 0.5),
                'Recall', recall_score(label, score > 0.5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Citeseer', help='dataset name: Citeseer/Pubmed/Arxiv/Children/CitationV8/Computers/History/Photo')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')

    parser.add_argument('--context_length', type=int, default=128)

    args = parser.parse_args()

    train_dominant(args)