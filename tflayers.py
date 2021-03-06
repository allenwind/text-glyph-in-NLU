import tensorflow as tf
from tensorflow.keras.layers import *

class MaskGlobalMaxPooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x = inputs
        x = x - (1 - mask) * 1e12 # 用一个大的负数mask
        x = tf.reduce_max(x, axis=1, keepdims=True)
        ws = tf.where(inputs == x, x, 0.0)
        ws = tf.reduce_sum(ws, axis=2)
        x = tf.squeeze(x, axis=1)
        return x, ws

class MaskGlobalMaxPooling2D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            # (batch_size, seqlen, 1, 1, 1)
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)

        x = inputs # (batch_size, seqlen, size, size, hdims)
        x = x - (1 - mask) * 1e12 # 用一个大的负数mask
        x = tf.reduce_max(x, axis=[2, 3], keepdims=True)
        ws = tf.where(inputs == x, x, 0.0)
        ws = tf.reduce_sum(ws, axis=2)
        x = tf.squeeze(x, axis=1)
        return x, ws

class AttentionPooling1D(tf.keras.layers.Layer):

    def __init__(self, hdims, kernel_initializer="glorot_uniform", **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.hdims = hdims
        self.kernel_initializer = tf.keras.initializers.get(
            kernel_initializer
        )
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = tf.keras.layers.Dense(
            units=self.hdims,
            kernel_initializer=self.kernel_initializer,
            # kernel_regularizer="l2",
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = tf.keras.layers.Dense(
            units=1,
            # kernel_regularizer="l1", # 添加稀疏性
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x0 = inputs
        # 计算每个 time steps 权重
        w = self.k_dense(inputs)
        w = self.o_dense(w)
        # 处理 mask
        w = w - (1 - mask) * 1e12
        # 权重归一化
        w = tf.math.softmax(w, axis=1) # 有mask位置对应的权重变为很小的值
        # 加权平均
        x = tf.reduce_sum(w * x0, axis=1)
        return x, w
