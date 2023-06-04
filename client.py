from tensorflow import keras
import tensorflow as tf
from metrics import *
from layers import ConcatenationAggregator, GraphConvolution, AttentionLayer
from utils import preprocess_adj, preprocess_feature


class Client(keras.Model):
    def __init__(self, args, adj_list, features, X_train, X_test, y):
        super().__init__()
        self.adj_list = adj_list
        self.features = features
        self.X_train = X_train
        self.X_test = X_test
        self.y = y
        self.reviews_num = args.reviews_num
        self.input_dim_r = args.input_dim_r
        self.input_dim_u = args.input_dim_u
        self.input_dim_i = args.input_dim_i
        self.class_size = args.class_size
        self.output_dim1 = args.output_dim1
        self.output_dim2 = args.output_dim2
        self.dropout = args.dropout

        self.r_agg_layer = ConcatenationAggregator(input_dim=self.input_dim_r + self.input_dim_u + self.input_dim_i,
                                                   output_dim=self.output_dim1)
        self.r_gcn_layer = GraphConvolution(input_dim=self.output_dim1,
                                            output_dim=self.output_dim2,
                                            activation=tf.nn.relu,
                                            dropout=self.dropout,
                                            is_sparse_inputs=True,
                                            norm=True)
        self.x_init = tf.keras.initializers.GlorotUniform()
        self.u = tf.Variable(initial_value=self.x_init(shape=(self.output_dim1 + self.output_dim2, self.class_size), dtype=tf.float32),
                             trainable=True)

    def train(self):
        support = self.adj_list
        features = []
        features.append(tf.convert_to_tensor(self.features[0], dtype=tf.float32))
        features.append(tf.convert_to_tensor(self.features[1], dtype=tf.float32))
        features.append(tf.convert_to_tensor(self.features[2], dtype=tf.float32))
        label = tf.convert_to_tensor(self.y, dtype=tf.float32)
        # idx_mask = [self.X_train, self.X_test]
        idx_mask = self.X_train
        supports = []
        for i in range(len(support[6])):
            hidden = [tf.cast(tf.SparseTensor(*preprocess_adj(support[6][i])), dtype=tf.float32)]
            supports.append(hidden)
        h_z = self.r_agg_layer((support, features))  # graphsage的结果，一个EagerTensor
        x = preprocess_feature(h_z)  # 把graphsage的结果变成了tuple
        num_features_nonzero = x[1].shape  # 特征里非零元素的个数
        x = tf.SparseTensor(*x)  # 把tuple变成了sparestensor
        outputs = []
        for i in range(len(supports)):
            gcn_out = self.r_gcn_layer((x, supports[i]), num_features_nonzero=num_features_nonzero, training=True)
            outputs.append(gcn_out)
        outputs = tf.reshape(outputs, [len(support[6]), self.reviews_num * self.output_dim2])
        outputs = AttentionLayer.attention(inputs=outputs, attention_size=len(support[6]), v_type='tanh')
        outputs = tf.reshape(outputs, [self.reviews_num, self.output_dim2])
        h_r = tf.concat([h_z, outputs], axis=1)
        # get masked data
        masked_data = tf.gather(h_r, idx_mask)
        masked_label = tf.gather(label, idx_mask)
        logits = tf.nn.softmax(tf.matmul(masked_data, self.u))
        loss = -tf.reduce_sum(tf.math.log(tf.nn.sigmoid(masked_label * logits)))

        acc = accuracy(logits, masked_label)
        rec = recall(logits, masked_label)
        f1 = (2 * acc * rec) / (acc + rec)
        return loss, acc, rec, f1

    def test(self):
        support = self.adj_list
        features = []
        features.append(tf.convert_to_tensor(self.features[0], dtype=tf.float32))
        features.append(tf.convert_to_tensor(self.features[1], dtype=tf.float32))
        features.append(tf.convert_to_tensor(self.features[2], dtype=tf.float32))
        label = tf.convert_to_tensor(self.y, dtype=tf.float32)
        # idx_mask = [self.X_train, self.X_test]
        idx_mask = self.X_test
        supports = []
        for i in range(len(support[6])):
            hidden = [tf.cast(tf.SparseTensor(*preprocess_adj(support[6][i])), dtype=tf.float32)]
            supports.append(hidden)
        h_z = self.r_agg_layer((support, features))  # graphsage的结果，一个EagerTensor
        x = preprocess_feature(h_z)  # 把graphsage的结果变成了tuple
        num_features_nonzero = x[1].shape  # 特征里非零元素的个数
        x = tf.SparseTensor(*x)  # 把tuple变成了sparestensor
        outputs = []
        for i in range(len(supports)):
            gcn_out = self.r_gcn_layer((x, supports[i]), num_features_nonzero=num_features_nonzero, training=True)
            outputs.append(gcn_out)
        outputs = tf.reshape(outputs, [len(support[6]), self.reviews_num * self.output_dim2])
        outputs = AttentionLayer.attention(inputs=outputs, attention_size=len(support[6]), v_type='tanh')
        outputs = tf.reshape(outputs, [self.reviews_num, self.output_dim2])
        h_r = tf.concat([h_z, outputs], axis=1)
        # get masked data
        masked_data = tf.gather(h_r, idx_mask)
        masked_label = tf.gather(label, idx_mask)
        logits = tf.nn.softmax(tf.matmul(masked_data, self.u))
        loss = -tf.reduce_sum(tf.math.log(tf.nn.sigmoid(masked_label * logits)))

        acc = accuracy(logits, masked_label)
        rec = recall(logits, masked_label)
        f1 = (2 * acc * rec) / (acc + rec)
        return loss, acc, rec, f1
