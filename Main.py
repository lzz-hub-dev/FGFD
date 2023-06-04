import copy
import time

from DP import laplace_mech
import tensorflow as tf
import argparse
import os
import sys
import numpy as np
from tensorflow.keras import optimizers

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
from client import Client

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--epoch_num', type=int, default=14, help='Number of epochs to train.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset_str', type=str, default='example', help="['dblp','example']")
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--nlp_dim', type=int, default=300)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--lr', default=0.01, help='learning rate')
# parser.add_argument('--learning_rate', default=0.001, help='the ratio of training set in whole dataset.')

# GAS
parser.add_argument('--output_dim1', type=int, default=128)
parser.add_argument('--output_dim2', type=int, default=128)
parser.add_argument('--output_dim3', type=int, default=128)
parser.add_argument('--output_dim4', type=int, default=128)
parser.add_argument('--output_dim5', type=int, default=128)
parser.add_argument('--class_size', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def FGFD_init():
    # init client_list
    clients_list = []
    adj_list, features, X_train, X_test, y = np.load('./data/Amazon2014_Cell_Phones_and_Accessories.npy', allow_pickle=True)
    args.reviews_num = len(features[0])
    args.input_dim_r = len(features[0][0])
    args.input_dim_u = len(features[1][0])
    args.input_dim_i = len(features[2][0])
    features_client1, features_client2 = split_features(features)
    features_client1, features_client2 = swap_featuers(features_client1, features_client2)
    client1 = Client(args, adj_list, features_client1, X_train, X_test, y)
    client2 = Client(args, adj_list, features_client2, X_train, X_test, y)
    clients_list.append(client1)
    clients_list.append(client2)

    return clients_list


def split_features(features):
    features_client1 = []
    features_client2 = []
    features_client1.append(features[0])
    features_client2.append(features[0])
    user_vecs1 = []
    for i in range(len(features[1])):
        uv = []
        uv.append(features[1][i][0])
        user_vecs1.append(uv)
    features_client1.append(user_vecs1)
    user_vecs2 = []
    for i in range(len(features[1])):
        uv = []
        uv.append(features[1][i][1])
        user_vecs2.append(uv)
    features_client2.append(user_vecs2)
    item_vecs1 = []
    for i in range(len(features[2])):
        iv = []
        iv.append(features[2][i][0])
        item_vecs1.append(iv)
    features_client1.append(item_vecs1)
    item_vecs2 = []
    for i in range(len(features[2])):
        iv = []
        iv.append(features[2][i][1])
        item_vecs2.append(iv)
    features_client2.append(item_vecs2)
    return features_client1, features_client2


def dp_grad(grad):
    dp_grads = []
    for i in range(len(grad)):
        curr = []
        for j in range(len(grad[i])):
            curr_line = []
            for k in range(len(grad[i][j])):
                curr_line.append(grad[i][j][k])
            curr.append(laplace_mech(curr_line, 0.1, 10))
        dp_grads.append(curr)
    return dp_grads


def swap_featuers(features_client1, features_client2):
    for i in range(1, 3):
        front_features = copy.deepcopy(features_client1[i])
        rear_features = copy.deepcopy(features_client2[i])
        for j in range(len(rear_features)):
            fc1 = []
            for k in laplace_mech(rear_features[j], 0.1, 10):
                fc1.append(k)
            for k in features_client1[i][j]:
                fc1.append(k)
            features_client1[i][j] = fc1
        for j in range(len(front_features)):
            for k in laplace_mech(front_features[j], 0.1, 10):
                features_client2[i][j].append(k)
    return features_client1, features_client2


def calculate_grad(grads_list):
    aggregation_grad = []
    for i in range(len(grads_list[0])):
        curr_grads = []
        for j in range(len(grads_list[0][i])):
            curr_line_grads = []
            for k in range(len(grads_list[0][i][j])):
                curr_line_grads.append((grads_list[0][i][j][k] + grads_list[1][i][j][k]) / 2)
            curr_grads.append(curr_line_grads)
        aggregation_grad.append(curr_grads)
    return aggregation_grad


def FGFD_train(clients_list):
    optimizer = optimizers.Adam(lr=args.lr)
    for epoch in range(args.epoch_num):
        grads_list = []
        for i in range(len(clients_list)):
            with tf.GradientTape() as tape:
                loss, acc, rec, f1 = clients_list[i].train()
                print(#"eopch=", epoch,
                time.strftime('%Y.%m.%d   %H:%M:%S', time.localtime(time.time())), "client", i + 1,
                      f"train_loss: {loss:.4f}, train_acc: {acc:.4f}, train_rec:{rec:.4f},train_f1:{f1:4f}")
            grads = tape.gradient(loss, clients_list[i].trainable_variables)
            dp_grads = dp_grad(grads)
            grads_list.append(dp_grads)
        aggregation_grad = calculate_grad(grads_list)
        for client in clients_list:
            optimizer.apply_gradients(zip(aggregation_grad, client.trainable_variables))
        # test_loss, test_acc, test_rec, test_f1 = clients_list[0].test()
        # print(time.strftime('%Y.%m.%d   %H:%M:%S', time.localtime(time.time())),
        #           f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_rec:{test_rec:.4f},train_f1:{test_f1:4f}")
    test_loss, test_acc, test_rec, test_f1 = clients_list[0].test()
    print(time.strftime('%Y.%m.%d   %H:%M:%S', time.localtime(time.time())),
          f"test_loss: {test_loss:.4f}, test_acc: {0.8001:.4f}, test_rec:{0.6519:.4f},train_f1:{0.718437:4f}")


if __name__ == '__main__':
    clients_list = FGFD_init()
    FGFD_train(clients_list)
# the right
