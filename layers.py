'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import tensorflow as tf
from tensorflow.keras import layers

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

'''Code about GCN is adapted from tkipf/gcn.'''


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / (1 - rate))


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def _call(self, inputs, adj_info):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(layers.Layer):
    """
    Graph convolution layer.
    Source: https://github.com/dragen1860/GCN-TF2/blob/89a71486b28a913fe50a69306f96de567a8c8bf8/layers.py#L95
    """

    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 is_sparse_inputs=False,
                 activation=tf.nn.relu,
                 norm=False,
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.norm = norm

        self.weights_ = []
        for i in range(1):
            w = self.add_variable('weight' + str(i), [input_dim, output_dim])
            self.weights_.append(w)
        if self.bias:
            self.bias = self.add_variable('bias', [output_dim])

    def call(self, inputs, num_features_nonzero, training=None):
        x, support_ = inputs

        # dropout
        if training is not False and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, num_features_nonzero)
        elif training is not False:
            x = tf.nn.dropout(x, self.dropout)

        # convolve
        supports = list()
        for i in range(len(support_)):
            if not self.featureless:
                pre_sup = dot(x, self.weights_[i], sparse=self.is_sparse_inputs)
            else:

                pre_sup = self.weights_[i]
            support = dot(support_[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        axis = list(range(len(output.get_shape()) - 1))
        mean, variance = tf.nn.moments(output, axis)
        scale = None
        offset = None
        variance_epsilon = 0.001
        output = tf.nn.batch_normalization(output, mean, variance, offset, scale,
                                           variance_epsilon)

        # bias
        if self.bias:
            output += self.bias
        if self.norm:
            return tf.nn.l2_normalize(self.activation(output), axis=None, epsilon=1e-12)

        return self.activation(output)


class ConcatenationAggregator(layers.Layer):
    """This layer equals to the equation (3) in
    paper 'Spam Review Detection with Graph Convolutional Networks.'
    """

    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu,
                 concat=False, **kwargs):
        super(ConcatenationAggregator, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.concat = concat
        self.con_agg_weights = self.add_weight('con_agg_weights', [input_dim, output_dim], dtype=tf.float32)
        # con_agg_weights是一个(19,64)的权重矩阵

    def __call__(self, inputs):
        adj_list, features = inputs

        review_vecs = tf.nn.dropout(features[0], self.dropout)
        user_vecs = tf.nn.dropout(features[1], self.dropout)
        item_vecs = tf.nn.dropout(features[2], self.dropout)

        # neighbor sample
        ri = tf.nn.embedding_lookup(item_vecs, tf.cast(adj_list[5], dtype=tf.int32))  # 从ri表中选取review对应的item的特征
        ri = tf.transpose(tf.random.shuffle(tf.transpose(ri)))

        ru = tf.nn.embedding_lookup(user_vecs, tf.cast(adj_list[4], dtype=tf.int32))  # 从ru表中选取review对应的user的特征
        ru = tf.transpose(tf.random.shuffle(tf.transpose(ru)))

        concate_vecs = tf.concat([review_vecs, ru, ri], axis=1)  # 一个(7,19)的特征矩阵

        # [nodes] x [out_dim]
        output = dot(concate_vecs, self.con_agg_weights, sparse=False)

        return self.act(output)


class AttentionLayer(layers.Layer):
    """ AttentionLayer is a function f : hkey × Hval → hval which maps
    a feature vector hkey and the set of candidates’ feature vectors
    Hval to an weighted sum of elements in Hval.
    """

    def attention(inputs, attention_size, v_type=None, return_weights=False, bias=True, joint_type='weighted_sum',
                  multi_view=True):  # 感觉这个attention方法是用在不同的元路径（构成的多张图）之间的，而不是用在一张图上不同节点之间的
        # 里面的两次expand_dims是为了最后reduce_sum加的吧
        if multi_view:
            inputs = tf.expand_dims(inputs, 0)  # 给inputs增加了一个维度，原本是(1,16228)变成了(1,1,16228)
        hidden_size = inputs.shape[-1]  # 16228

        # Trainable parameters
        w_omega = tf.Variable(tf.random.uniform([hidden_size, attention_size]))  # (16228,1)的可训练矩阵,值在默认(0,1)之间
        b_omega = tf.Variable(tf.random.uniform([attention_size]))  # (1,)的待训练参数
        u_omega = tf.Variable(tf.random.uniform([attention_size]))  # (1,)的待训练参数，表示注意力？

        v = tf.tensordot(inputs, w_omega, axes=1)  # inputs先乘以权重矩阵，得到一个值，维度是(1,1,1)
        if bias is True:
            v += b_omega
        if v_type is 'tanh':
            v = tf.tanh(v)
        if v_type is 'relu':
            v = tf.nn.relu(v)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # 再乘以另外一个值，得到的维度是(1,1)
        weights = tf.nn.softmax(vu, name='alphas')  # softmax一下，维度还是(1,1)

        if joint_type is 'weighted_sum':
            output = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), 1)
        if joint_type is 'concatenation':
            output = tf.concat(inputs * tf.expand_dims(weights, -1), 2)
        if not return_weights:
            return output
        else:
            return output, weights
