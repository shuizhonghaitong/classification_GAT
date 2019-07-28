import pickle
import random
import time
import networkx as nx
import numpy as np
import tensorflow.contrib.keras as kr
import scipy.sparse as sp
from Parameters import Parameters as pm
from utils import *


class Data(object):
    def __init__(self):
        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []

        self.train_ids = []
        self.dev_ids = []
        self.test_ids = []

        self.train_labels = []
        self.dev_labels = []
        self.test_labels = []

        self.train_label_ids = []
        self.dev_label_ids = []
        self.test_label_ids = []

        self.train_masks = []
        self.dev_masks = []
        self.test_masks = []

    def load_network(self):
        print('start loading network...')
        start_time = time.time()
        with open(pm.network_filename, 'rb') as f:
            graph = pickle.load(f)
        pm.nb_nodes = len(graph)
        # pm.ft_size = pm.nb_nodes

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        self.biases = preprocess_adj_bias(adj)

        # row = []
        # col = []
        # data = []
        # for i in range(pm.nb_nodes):
        #     row.append(i)
        #     col.append(i)
        #     data.append(1)
        # features = sp.csr_matrix((data, (row, col)), shape=(pm.nb_nodes, pm.nb_nodes))
        features=[]
        for i in range(pm.nb_nodes):
            features.append(np.empty((1,256)))
        features=np.concatenate(features,0)
        features = preprocess_features(features)
        self.features = features[np.newaxis]
        print('loading network finished! Having used {:.4}s.'.format(time.time() - start_time))

    def load_data(self):
        print('start loading data...')
        start_time = time.time()
        with open(pm.train_filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                label, content, mentions = line.strip().split('\t')
                self.train_texts.append(mentions.split('#'))
                self.train_labels.append(label)
        with open(pm.val_filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                label, content, mentions = line.strip().split('\t')
                self.dev_texts.append(mentions.split('#'))
                self.dev_labels.append(label)
        with open(pm.test_filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                label, content, mentions = line.strip().split('\t')
                self.test_texts.append(mentions.split('#'))
                self.test_labels.append(label)

        mention2id = read_mention2id_dict()
        for text in self.train_texts:
            self.train_ids.append([mention2id[word] for word in text])
        for text in self.dev_texts:
            self.dev_ids.append([mention2id[word] for word in text])
        for text in self.test_texts:
            self.test_ids.append([mention2id[word] for word in text])

        categories, cat_to_id = read_category()
        for label in self.train_labels:
            self.train_label_ids.append(cat_to_id[label])
        for label in self.dev_labels:
            self.dev_label_ids.append(cat_to_id[label])
        for label in self.test_labels:
            self.test_label_ids.append(cat_to_id[label])

        # mask
        for i in range(len(self.train_ids)):
            length = len(self.train_ids[i])
            if length <= pm.seq_length:
                self.train_masks.append([1] * length + [0] * (pm.seq_length - length))
            else:
                self.train_masks.append([1] * pm.seq_length)
        self.train_masks = np.array(self.train_masks, dtype=np.float32)

        for i in range(len(self.dev_ids)):
            length = len(self.dev_ids[i])
            if length <= pm.seq_length:
                self.dev_masks.append([1] * length + [0] * (pm.seq_length - length))
            else:
                self.dev_masks.append([1] * pm.seq_length)
        self.dev_masks = np.array(self.dev_masks, dtype=np.float32)

        for i in range(len(self.test_ids)):
            length = len(self.test_ids[i])
            if length <= pm.seq_length:
                self.test_masks.append([1] * length + [0] * (pm.seq_length - length))
            else:
                self.test_masks.append([1] * pm.seq_length)
        self.test_masks = np.array(self.test_masks, dtype=np.float32)

        # padding x
        self.train_ids = kr.preprocessing.sequence.pad_sequences(self.train_ids, pm.seq_length, padding='post',
                                                                 truncating='post', value=-1)
        self.dev_ids = kr.preprocessing.sequence.pad_sequences(self.dev_ids, pm.seq_length, padding='post',
                                                               truncating='post', value=-1)
        self.test_ids = kr.preprocessing.sequence.pad_sequences(self.test_ids, pm.seq_length, padding='post',
                                                                truncating='post', value=-1)
        # categorical y
        self.train_label_ids = kr.utils.to_categorical(self.train_label_ids)
        self.dev_label_ids = kr.utils.to_categorical(self.dev_label_ids)
        self.test_label_ids = kr.utils.to_categorical(self.test_label_ids)

        print('loading data finished! Having used {:.4}s.'.format(time.time() - start_time))

    def shuffle_train_data(self):
        shuffled_index = list(range(len(self.train_labels)))
        random.shuffle(shuffled_index)
        self.train_texts = [self.train_texts[index] for index in shuffled_index]
        self.train_ids = np.array([self.train_ids[index] for index in shuffled_index])
        self.train_labels = [self.train_labels[index] for index in shuffled_index]
        self.train_label_ids = np.array([self.train_label_ids[index] for index in shuffled_index])
        self.train_masks = np.array([self.train_masks[index] for index in shuffled_index])
