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
from imagefont import char_to_glyph

# result in THUCNews
# 14632/14632 985s 67ms/step 1 epochs
# - loss: 1.0370 - accuracy: 0.6737 
# - val_loss: 0.7873 - val_accuracy: 0.7496
# - test_loss: 0.7999 - test_accuracy: 0.7476

class GlyceEmbedding(tf.keras.layers.Layer):

    def __init__(self, char2id, maxlen, start=2, size=32, flatten=True, as_image=True, **kwargs):
        super(GlyceEmbedding, self).__init__(**kwargs)
        self.size = size
        self.maxlen = maxlen
        # 汉字偏平输出，即句子合并到一张图上
        # 这个参数影响输出形状
        self.flatten = flatten
        self.as_image = as_image
        shape = (len(char2id)+start, size, size, 1)
        # zeros for UNK and MASK
        self.embeddings = np.zeros(shape)
        for c, i in char2id.items():
            self.embeddings[i] = char_to_glyph(c)
        self.embeddings = tf.cast(self.embeddings, dtype=tf.float32)

    def call(self, inputs, mask=None):
        # (batch_size, seq_len, size_1, size_2, channels)
        x = tf.gather(self.embeddings, inputs)
        if not self.flatten:
            return x
        x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
        seq_len = tf.shape(inputs)[1]
        # (batch_size, size_1, seq_len * size_2, channels)
        x = tf.reshape(x, (-1, self.size, self.maxlen * self.size, 1))
        return x

# 处理数据
X, y, classes = load_THUCNews_title_label()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=732
)

num_classes = len(classes)
# 转化成字id
tokenizer = Tokenizer()
tokenizer.fit(X_train)

def create_dataset(X, y, maxlen=48):
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
X_test, y_test = create_dataset(X_test, y_test)

# 模型
num_words = len(tokenizer)
embedding_dims = 128

inputs = Input(shape=(maxlen,), dtype=tf.int32)
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)

gembedding = GlyceEmbedding(tokenizer.char2id, maxlen, flatten=True)
conv1 = Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 2),
    activation="relu",
    padding="same",
)
conv2 = Conv2D(
    filters=128,
    kernel_size=(5, 5),
    strides=(1, 3),
    activation="relu",
    padding="same",
)
pool2d = GlobalMaxPool2D()

x = gembedding(inputs)

view_text_glyph = Model(inputs, x)

x = conv1(x)
x = conv2(x)
x = pool2d(x)

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
epochs = 10
model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)


# 评估
model.evaluate(X_test, y_test)
