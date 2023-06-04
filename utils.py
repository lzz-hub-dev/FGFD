import scipy.sparse as sp
import numpy as np


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """
    Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_feature(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """

    features = sp.lil_matrix(features)
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    return sparse_to_tuple(features)


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.int32)


def pad_adjlist(x_data):
    # Get lengths of each row of data
    lens = np.array([len(x_data[i]) for i in range(len(x_data))])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    padded = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        padded[i] = np.random.choice(x_data[i], mask.shape[1])
    padded[mask] = np.hstack((x_data[:]))
    return padded


def matrix_to_adjlist(M, pad=True):
    adjlist = []
    for i in range(len(M)):
        adjline = [i]
        for j in range(len(M[i])):
            if M[i][j] == 1:
                adjline.append(j)
        adjlist.append(adjline)
    if pad:
        adjlist = pad_adjlist(adjlist)
    return adjlist


def adjlist_to_matrix(adjlist):
    nodes = len(adjlist)
    M = np.zeros((nodes, nodes))
    for i in range(nodes):
        for j in adjlist[i]:
            M[i][j] = 1
    return M


def pairs_to_matrix(pairs, nodes):
    M = np.zeros((nodes, nodes))
    for i, j in pairs:
        M[i][j] = 1
    return M