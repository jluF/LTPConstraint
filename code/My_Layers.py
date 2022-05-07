import tensorflow as tf
import keras
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import math
from keras.layers import Dropout, Input, Dense, Activation, BatchNormalization, Conv2D, Embedding, LSTM
from keras.initializers import Constant, RandomUniform
from keras.utils import to_categorical
from keras.backend import softsign, sigmoid, relu, sign, tanh
from scipy.sparse import diags
class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i - i % 2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])  # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])  # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"model_dim": self._model_dim}
        base_config = super(PositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ScaledDotProductAttention(Layer):

    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = -2 ** 32 + 1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs + masks * self._masking_num
        return outputs

    def future_mask(self, inputs):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * self._masking_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

    def call(self, inputs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')

        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
        if self._masking:
            scaled_matmul = self.mask(scaled_matmul, masks)  # Mask(opt.)

        if self._future:
            scaled_matmul = self.future_mask(scaled_matmul)

        softmax_out = K.softmax(scaled_matmul)  # SoftMax
        # Dropout
        out = K.dropout(softmax_out, self._dropout_rate)

        outputs = K.batch_dot(out, values)

        return outputs

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"masking": self._masking,
                  "future": self._future,
                  "dropout_rate": self._dropout_rate
                  }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class MultiHeadAttention(Layer):

    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=True, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._masking = masking
        self._future = future
        self._trainable = trainable
        self._weights_queries = None
        self._weights_keys = None
        self._weights_values = None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        queries_linear = K.dot(queries, self._weights_queries)
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)
        # batch_size, seq_size, self._n_heads*self._head_dim
        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)

        if self._masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]

        attention = ScaledDotProductAttention(
            masking=self._masking, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)

        return outputs

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"n_heads": self._n_heads,
                  "head_dim": self._head_dim,
                  "dropout_rate": self._dropout_rate,
                  "masking": self._masking,
                  "future": self._future,
                  "trainable": self._trainable
                  }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class PositionWiseFeedForward(Layer):

    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")
        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")
        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        return outputs

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"model_dim": self._model_dim,
                  "inner_dim": self._inner_dim,
                  "trainable": self._trainable
                  }
        base_config = super(PositionWiseFeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"epsilon": self._epsilon}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConstraintAndResult(Layer):
    def __init__(self, seq_len, steps, **kwargs):
        self._seq_len = seq_len
        self.steps = steps
        super(ConstraintAndResult, self).__init__(**kwargs)

    def build(self, input_shape):
        self.s = self.add_weight(
            shape=(1,),
            initializer=Constant(value=0.5),
            trainable=True,
            name='s')
        self.w = self.add_weight(
            shape=(1,),
            initializer=RandomUniform(minval=0, maxval=1, seed=None),
            trainable=True,
            name='w')
        self.rho_p = self.add_weight(
            shape=(self._seq_len, self._seq_len),
            initializer=RandomUniform(minval=0, maxval=1, seed=None),
            trainable=True,
            name='rho_p'
        )
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=Constant(value=0.005),
            trainable=True,
            name='alpha'
        )
        self.belt = self.add_weight(
            shape=(1,),
            initializer=Constant(value=0.05),
            trainable=True,
            name='belt'
        )
        self.lr_alpha = self.add_weight(
            shape=(1,),
            initializer=Constant(value=0.99),
            trainable=True,
            name='lr_alpha'
        )
        self.lr_belt = self.add_weight(
            shape=(1,),
            initializer=Constant(value=0.99),
            trainable=True,
            name='lr_belt'
        )
        super(ConstraintAndResult, self).build(input_shape)

    def T_A(self, inputs, M):
        return (inputs + tf.transpose(inputs, (0, 2, 1))) / 2 * M

    def PPcell(self, U, M, Lm, A, A_hat, t):
        G = U / 2 - K.repeat_elements(
            tf.expand_dims(Lm * sign(tf.reduce_sum(A, -1) - 1), 2),
            self._seq_len,
            axis=2
        )
        A_hat = A_hat + self.alpha * tf.pow(self.lr_alpha, t) * A_hat * M * (G + tf.transpose(G, (0, 2, 1)))
        A_hat = relu(tf.abs(A_hat) - self.rho_p * self.alpha * tf.pow(self.lr_alpha, t))
        A_hat = 1 - relu(1 - A_hat)
        A = self.T_A(A_hat, M)
        Lm = Lm + self.belt * tf.pow(self.lr_belt, t) * relu(tf.reduce_sum(A, -1) - 1)
        return Lm, A, A_hat

    def call(self, scores, M):
        # 初始化
        U = scores-self.s
        A_hat = scores
        A = self.T_A(A_hat, M)
        Lm = self.w * (relu(tf.reduce_sum(A, -1) - 1))
        A_list = list()
        # 梯度下降
        for t in range(self.steps):
            Lm, A, A_hat = self.PPcell(U, M, Lm, A, A_hat, t)
            A_list.append(A)
        return A_list[-1]

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {
            "seq_len": self._seq_len,
            "steps": self.steps
                  }
        base_config = super(ConstraintAndResult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(
          Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(Dropout(0.2))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(
        keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(Dropout(0.2))
    result.add(BatchNormalization())

    if apply_dropout:
        result.add(keras.layers.Dropout(0.5))

    result.add(keras.layers.ReLU())

    return result

def Generator(inputs, seq_len, input_channel, output_channel):

    down_stack = [
        # downsample(16, 4, apply_batchnorm=False),  # (bs, 256, 256, 16)
        # downsample(32, 4), # (bs, 128, 128, 32)
        downsample(64, 4), # (bs, 64, 64, 64)
        downsample(128, 4), # (bs, 32, 32, 128)
        downsample(256, 4), # (bs, 16, 16, 256)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 512)+(bs, 2, 2, 512)=(bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(256, 4), # (bs, 16, 16, 512)
        upsample(128, 4), # (bs, 32, 32, 256)
        upsample(64, 4), # (bs, 64, 64, 128)
        # upsample(32, 4), # (bs, 128, 128, 64)
        # upsample(16, 4), # (bs, 256, 256, 32)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = keras.layers.Conv2DTranspose(output_channel, 4, strides=2, padding='same', kernel_initializer=initializer, activation="sigmoid") # (bs, 512, 512, 1)

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)  # 第一个x是(bs, 2, 2, 512)
        x = keras.layers.Concatenate()([x, skip])

    x = last(x)

    return x


def Discriminator(inp, tar, seq_len, input_channel, output_channel):
    initializer = tf.random_normal_initializer(0., 0.02)

    x = keras.layers.concatenate([inp, tar]) # (bs, 256, 256, input_channel+outputchannel)

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    # 二维矩阵的上下左右各添加一个0的填充层
    zero_pad1 = keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = BatchNormalization()(conv)

    leaky_relu = keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = Conv2D(1, 4, strides=1, activation='sigmoid', kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return last

# def constraint_matrix_batch(x):
#     # size: None, seq_len, 5
#     # 约束 1
#     c = to_categorical(x, num_classes=5, dtype='int32')
#     base_a = c[:, :, 1]
#     base_u = c[:, :, 2]
#     base_g = c[:, :, 3]
#     base_c = c[:, :, 4]
#     batch = base_a.shape[0]
#     length = base_a.shape[1]
#     au = tf.matmul(tf.expand_dims(base_a, -1), tf.expand_dims(base_u, 1))
#     au_ua = au + tf.transpose(au, (0, 2, 1))
#     cg = tf.matmul(tf.expand_dims(base_c, -1), tf.expand_dims(base_g, 1))
#     cg_gc = cg + tf.transpose(cg, (0, 2, 1))
#     ug = tf.matmul(tf.expand_dims(base_u, -1), tf.expand_dims(base_g, 1))
#     ug_gu = ug + tf.transpose(ug, (0, 2, 1))
#     M = au_ua + cg_gc + ug_gu
#     # 约束 2
#     mask = diags([1] * 7, [-3, -2, -1, 0, 1, 2, 3], shape=(M.shape[-2], M.shape[-1])).toarray()
#     mask = tf.convert_to_tensor(1-mask, tf.int32, name = 't')
#     M = M * mask
#     return M

def constraint_matrix_batch(x):
    # size: None, seq_len, 5
    # 约束 1
    c = to_categorical(x, num_classes=5, dtype='int32')
    base_a = c[:, :, 1]
    base_u = c[:, :, 2]
    base_g = c[:, :, 3]
    base_c = c[:, :, 4]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = np.matmul(np.expand_dims(base_a, -1), np.expand_dims(base_u, 1))
    au_ua = au + np.transpose(au, (0, 2, 1))
    cg = np.matmul(np.expand_dims(base_c, -1), np.expand_dims(base_g, 1))
    cg_gc = cg + np.transpose(cg, (0, 2, 1))
    ug = np.matmul(np.expand_dims(base_u, -1), np.expand_dims(base_g, 1))
    ug_gu = ug + np.transpose(ug, (0, 2, 1))
    M = au_ua + cg_gc + ug_gu
    # 约束 2
    mask = diags([1] * 7, [-3, -2, -1, 0, 1, 2, 3], shape=(M.shape[-2], M.shape[-1])).toarray()
    mask = 1-mask
    M = M * mask
    return M



class ShapeOutputslayer(Layer):

    def __init__(self, seq_len, **kwargs):
        self.seq_len = seq_len
        super(ShapeOutputslayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.seq_len, self.seq_len),
            initializer='one',
            name='w')
        super(ShapeOutputslayer, self).build(input_shape)

    def call(self, inputs):
        pred, shape = inputs
        return sigmoid(pred + self.w * shape)

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"seq_len": self.seq_len}
        base_config = super(ShapeOutputslayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# crossentropy_loss disadvantage: needed a hand-tuned weight to balance
# 512 sequence length use weight=1456
# 128 sequence length use weight=256
def weighted_cross_entropy_with_logits_Me(labels, logits, pos_weight=1456):
    return tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight)

def F1_loss(y_true, y_pred):
    TP = tf.reduce_sum(y_true * y_pred)
    FP = tf.reduce_sum((1-y_true) * y_pred)
    FN = tf.reduce_sum(y_true * (1-y_pred))
    loss = -2 * TP / (2 * TP + FP + FN)
    return loss


# logistic loss: the loss of binary crossentropy
def logistic_regression_loss(y_true, y_pred):
    loss = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
    loss = tf.reduce_sum(loss) * (-1) / (128 * 128)
    return loss


# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 100)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 100)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return (-1) * K.sum(s, axis=0)


def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(sigmoid((y_pred-threshold) * 8.3e+6), 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / P
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(sigmoid((y_pred-threshold) * 8.3e+6), 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N

def pr_auc(y_true, y_pred):
    precisions = tf.stack([binary_precision(y_true, y_pred, k) for k in np.linspace(0, 1, 90)], axis=0)
    recalls = tf.stack([binary_recall(y_true, y_pred, k) for k in np.linspace(0, 1, 90)], axis=0)
    recalls = tf.concat([tf.ones((1,)), recalls], axis=0)
    binSizes = -(recalls[1:] - recalls[:-1])
    s = precisions * binSizes
    return (-1) * K.sum(s, axis=0)

def binary_precision(y_true, y_pred, threshold):
    y_pred = K.cast((relu(y_pred - threshold)+1), 'float32')
    TP = tf.reduce_sum(y_true * y_pred)
    FP = tf.reduce_sum((1-y_true) * y_pred)
    return TP / (TP + FP)

def binary_recall(y_true, y_pred, threshold):
    y_pred = K.cast(relu(y_pred - threshold), 'float32')
    TP = tf.reduce_sum(y_true * y_pred)
    FN = tf.reduce_sum(y_true*(1-y_pred))
    return TP / (TP + FN)

def val_test(y_true, y_pred):
    TP = tf.reduce_sum(y_true * y_pred)
    FP = tf.reduce_sum((1 - y_true) * y_pred)
    TN = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    FN = tf.reduce_sum(y_true * (1 - y_pred))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall, (TP, FP, TN, FN)

def precision(y_true, y_pred):
    TP = tf.reduce_sum(y_true * y_pred)
    FP = tf.reduce_sum((1 - y_true) * y_pred)
    return TP / (TP + FP)

def recall(y_true, y_pred):
    TP = tf.reduce_sum(y_true * y_pred)
    FN = tf.reduce_sum(y_true * (1 - y_pred))
    return TP / (TP + FN)