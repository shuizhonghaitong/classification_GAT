import tensorflow as tf

class Parameters(object):

    hid_units = [128, 64]  # 2层GAT：第1层每个head的输出维度是8，第2层每个head的输出维度是类别数
    n_heads = [1, 1]  # 2层GAT：第1层有8个head，第2层有1个head
    nonlinear=[tf.nn.elu,tf.nn.elu]
    residual=False
    attention_size = 32

    nb_nodes=-1
    ft_size=256

    seq_length=20
    num_classes=9

    learning_rate=0.005
    attn_drop=0.6
    ffd_drop=0.6

    num_epoches=100
    batch_size=64

    network_filename='data/tang.poetry.graph'
    # network_filename='data/ind.cora.graph'
    train_filename='data/big_tang.train.txt'
    val_filename='data/big_tang.val.txt'
    test_filename='data/big_tang.test.txt'