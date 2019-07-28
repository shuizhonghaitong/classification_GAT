import tensorflow as tf
from Parameters import Parameters as pm
from models.gat import GAT
from data import Data
from utils import compute_criterion, evaluate


def train():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    max_dev_f1 = -1.0
    max_epoch = -1
    best_test_acc, best_test_p, best_test_r, best_test_f1 = -1.0, -1.0, -1.0, -1.0
    for epoch_id in range(pm.num_epoches):
        data.shuffle_train_data()
        train_num = len(data.train_texts)
        num_batches = train_num // pm.batch_size + 1
        for batch_id in range(num_batches):
            start = batch_id * pm.batch_size
            end = (batch_id + 1) * pm.batch_size
            if end > train_num:
                end = train_num
            if start == end:
                break
            batch_x = data.train_ids[start:end]
            batch_y = data.train_label_ids[start:end]
            batch_mask = data.train_masks[start:end]
            feed_dict = model.feed_data(batch_x, batch_y, batch_mask, data.features, data.biases, pm.attn_drop,
                                        pm.ffd_drop)
            _,train_loss, y_pred = sess.run([model.train_op, model.loss, model.logits],
                                             feed_dict=feed_dict)
            train_acc,train_p, train_r, train_f1 = compute_criterion(y_pred, batch_y)
            if batch_id % 10 == 0:
                print('Epoch:{}/{} batch:{}/{} train_loss:{:.4} train_accuracy:{:.4} p:{:.4} r:{:.4} f1:{:.4}'.format(
                    (epoch_id + 1), pm.num_epoches, (batch_id + 1), num_batches, train_loss, train_acc, train_p,
                    train_r, train_f1))
            if batch_id==0:
                break
        dev_acc, dev_p, dev_r, dev_f1 = evaluate(sess, data, model, 'dev')
        print('Epoch:{}/{} dev_accuracy:{:.4} p:{:.4} r:{:.4} f1:{:.4}'.format(
            (epoch_id + 1), pm.num_epoches, dev_acc, dev_p, dev_r, dev_f1))
        test_acc, test_p, test_r, test_f1 = evaluate(sess, data, model, 'test')
        print('Epoch:{}/{} test_accuracy:{:.4} p:{:.4} r:{:.4} f1:{:.4}'.format(
            (epoch_id + 1), pm.num_epoches, test_acc, test_p, test_r, test_f1))
        print('-------------------------------------------------------------------------------')
        if dev_f1 > max_dev_f1:
            max_dev_f1 = dev_f1
            max_epoch = epoch_id
            best_test_acc = test_acc
            best_test_p = test_p
            best_test_r = test_r
            best_test_f1 = test_f1
    print('epoch:{} best val F1:{:.4}'.format((max_epoch + 1), max_dev_f1))
    print('best test acc:{:.4} p:{:.4} r:{:.4} f1:{:.4}'.format(best_test_acc, best_test_p, best_test_r, best_test_f1))


if __name__ == '__main__':
    pm = pm

    data = Data()
    data.load_network()
    data.load_data()

    model = GAT()
    train()
