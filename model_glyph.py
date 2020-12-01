import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

from dataset import load_THUCNews_title_label
from dataset import Tokenizer, find_best_maxlen
from imagefont import ImageFontTransformer
from tflayers import AttentionPooling1D

class GlyphEmbedding(tf.keras.layers.Layer):

    def __init__(self, char2id, maxlen, start=2, size=32, flatten=True, as_image=True, **kwargs):
        super(GlyphEmbedding, self).__init__(**kwargs)
        self.size = size
        self.maxlen = maxlen
        # 汉字偏平输出，即句子合并到一张图上
        # 这个参数影响输出形状
        self.flatten = flatten
        self.as_image = as_image
        shape = (len(char2id)+start, size, size, 1)
        # zeros for UNK and MASK
        tr = ImageFontTransformer()
        self.embeddings = np.zeros(shape)
        for c, i in char2id.items():
            self.embeddings[i] = tr.transform(c)
        self.embeddings = tf.cast(self.embeddings, dtype=tf.float32)

    def call(self, inputs, mask=None):
        # (batch_size, seq_len, size_1, size_2, channels)
        x = tf.gather(self.embeddings, inputs)
        if not self.flatten:
            return x
        x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
        seq_len = tf.shape(inputs)[1]
        # (batch_size, size_1, seq_len * size_2, 1)
        x = tf.reshape(x, (-1, self.size, self.maxlen * self.size, 1))
        return x

class ImageFlatten2D(tf.keras.layers.Layer):
    """(batch_size, seq_len, size_1, size_2, 1)
    => (batch_size, size_1, seq_len * size_2, 1)
    """

    def __init__(self, maxlen, size, **kwargs):
        super(ImageFlatten2D, self).__init__(**kwargs)
        self.size = size
        self.maxlen = maxlen

    def call(self, inputs):
        x = inputs
        x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
        seq_len = tf.shape(inputs)[1]
        x = tf.reshape(x, (-1, self.size, self.maxlen * self.size, 1))
        return x

# 处理数据
X, y, classes = load_THUCNews_title_label(limit=None)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=732
)

num_classes = len(classes)
# 转化成字id
tokenizer = Tokenizer()
tokenizer.fit(X_train)

maxlen = 32
def create_dataset(X, y, maxlen=maxlen):
    X = tokenizer.transform(X)
    y = tf.keras.utils.to_categorical(y)
    X = sequence.pad_sequences(
        X,
        maxlen=maxlen,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0.0
    )
    return X, y

# maxlen = find_best_maxlen(X_train, mode="max")
X_train, y_train = create_dataset(X_train, y_train)

# 模型
num_words = len(tokenizer)
embedding_dims = 128

inputs = Input(shape=(maxlen,), dtype=tf.int32)
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)

gembedding = GlyphEmbedding(tokenizer.char2id, maxlen, flatten=False)
conv1 = Conv2D(
    filters=64,
    kernel_size=(2, 2),
    strides=(1, 1),
    activation="relu",
    padding="same",
)
conv2 = Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation="relu",
    padding="same",
)
pool2d = Lambda(lambda x: tf.reduce_max(x, axis=[2, 3]))
apool = AttentionPooling1D(128)

# (batch_size, seq_len, size, size, 1)
x = gembedding(inputs)

vx = ImageFlatten2D(maxlen=48, size=32)(x)
view_text_glyph = Model(inputs, vx)
view_text_glyph.summary()

x = conv1(x)
x = conv2(x)
x = pool2d(x) # (batch_size, seq_len, hdims)
x, w = apool(x, mask=mask)
apool_outputs = Model(inputs, w)
apool_outputs.summary()

x = Dense(128)(x)
x = Dropout(0.2)(x)
x = Activation("relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.summary()

# 训练
batch_size = 32
epochs = 1
model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

model.save_weights("checkpoint.weights")

# 评估
X_etest, y_etest = create_dataset(X_test, y_test)
model.evaluate(X_etest, y_etest)

import matplotlib.pyplot as plt
from color import render_color_image

id_to_classes = {j:i for i,j in classes.items()}
def visualization():
    for sample, label in zip(X_test, y_test):
        sample_len = len(sample)
        if sample_len > maxlen:
            sample_len = maxlen

        x = np.array(tokenizer.transform([sample]))
        x = sequence.pad_sequences(
            x, 
            maxlen=maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0
        )

        y_pred = model.predict(x)[0]
        y_pred_id = np.argmax(y_pred)
        # 预测错误的样本跳过
        if y_pred_id != label:
            continue
            
        # 预测权重
        weights = apool_outputs.predict(x)[0]
        weights = weights.flatten()[:sample_len]
        # sample += " " * (maxlen - sample_len)
        # weights = list(weights)
        # weights.extend([0] * (maxlen - sample_len))
        image = render_color_image(sample, weights)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        # print(" =>", id_to_classes[y_pred_id])
        # input() # 按回车预测下一个样本

visualization()
