import pickle
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn import metrics
from Parameters import Parameters as pm

def read_mention2id_dict():
    # 注：id从1开始
    with open('data/tang.poetry.mention2id_dict','rb') as f:
        mention2id=pickle.load(f)
    return mention2id

def read_category():
    categories = ['战争', '送别', '闺怨', '咏物', '怀古', '思乡', '怀人', '田园', '山水']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def compute_criterion(y_pred,y):
    y_pred=np.argmax(y_pred,1)
    y=np.argmax(y,1)
    correct=np.equal(y_pred,y)
    acc=np.average(correct)
    pre=metrics.precision_score(y,y_pred,average='macro')
    rec=metrics.recall_score(y,y_pred,average='macro')
    f1=metrics.f1_score(y,y_pred,average='macro')
    return acc,pre,rec,f1

def evaluate(sess,data,model,name):
    if name=='dev':
        xs=data.dev_ids
        ys=data.dev_label_ids
        masks=data.dev_masks
    elif name=='test':
        xs=data.test_ids
        ys=data.test_label_ids
        masks=data.test_masks
    else:
        print("Error: wrong evaluate name,", name)
    sample_num=len(ys)
    num_batch=sample_num//pm.batch_size+1
    y_preds=[]
    for batch_id in range(num_batch):
        start = batch_id * pm.batch_size
        end = (batch_id + 1) * pm.batch_size
        if sample_num < end:
            end = sample_num
        if start == end:
            break
        batch_x = xs[start:end]
        batch_y = ys[start:end]
        batch_mask = masks[start:end]
        feed_dict = model.feed_data(batch_x, batch_y, batch_mask, data.features, data.biases, 0.0, 0.0)
        loss,y_pred=sess.run([model.loss,model.logits],feed_dict=feed_dict)
        y_preds.append(y_pred)
    y_preds=np.concatenate(y_preds,0)
    acc,pre,rec,f1=compute_criterion(y_preds,ys)
    return acc,pre,rec,f1

def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    # coords和values对照着看。coords:(49216,2) values:(49216) shape:(2708,1433)

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)) # (2708,1)
    rowsum=rowsum.astype('float32')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # 对features每行做归一化，即每行中的每个元素在0~1之间，且一行中的元素加起来为1.
    # return features.todense(), sparse_to_tuple(features)
    return features



