import re
import glob
import itertools
import random
import collections
import numpy as np
import pandas as pd

# 加载分类数据

def build_vocab(X):
    # 快速建立词表
    vocab = set(itertools.chain(*X))
    char2id = {c: i for i, c in enumerate(vocab, start=2)}
    return char2id

_SW = "/home/zhiwen/workspace/dataset/stopwords/stopwords.txt"
def load_stop_words(file=_SW):
    with open(file, "r") as fp:
        stopwords = fp.read().splitlines()
    return set(stopwords)

_THUCNews = "/home/zhiwen/workspace/dataset/THUCNews-title-label.txt"
def load_THUCNews_title_label(file=_THUCNews, nobrackets=True, limit=None):
    with open(file, encoding="utf-8") as fd:
        text = fd.read()
    lines = text.split("\n")[:-1]
    random.shuffle(lines)
    titles = []
    labels = []
    for line in lines[:limit]:
        title, label = line.split("\t")
        if not title:
            continue

        # 去掉括号内容
        if nobrackets:
            title = re.sub("\(.+?\)", lambda x: "", title)

        titles.append(title)
        labels.append(label)
    categoricals = list(set(labels))
    categoricals.sort()
    categoricals = {label: i for i, label in enumerate(categoricals)}
    clabels = [categoricals[i] for i in labels]
    return titles, clabels, categoricals

_taotiao_news = "/home/zhiwen/workspace/dataset/classification/taotiao-news-abc.txt"
def load_taotiao_news(file=_taotiao_news):
    with open(file, encoding="utf-8") as fd:
        text = fd.read()
    lines = text.split("\n")[:-1]
    random.shuffle(lines)
    titles = []
    labels = []
    for line in lines:
        title, tags, label = line.rsplit("\t", 2)
        if not title:
            continue

        titles.append(title)
        labels.append(label)
    categoricals = list(set(labels))
    categoricals.sort()
    categoricals = {label: i for i, label in enumerate(categoricals)}
    clabels = [categoricals[i] for i in labels]
    return titles, clabels, categoricals

_w100k = "/home/zhiwen/workspace/dataset/classification/weibo_senti_100k/weibo_senti_100k.csv"
def load_weibo_senti_100k(file=_w100k, noe=False):
    df = pd.read_csv(file)
    df = df.sample(frac=1) # shuffle
    X = df.review.to_list()
    y = df.label.to_list()
    # 去 emoji 表情，提升样本训练难度
    if noe:
        X = [re.sub("\[.+?\]", lambda x:"", s) for s in X]
    categoricals = {"负面": 0, "正面": 1}
    return X, y, categoricals

_MOODS = "/home/zhiwen/workspace/dataset/classification/simplifyweibo_4_moods.csv"
def load_simplifyweibo_4_moods(file=_MOODS):
    df = pd.read_csv(file)
    df = df.sample(frac=1) # shuffle
    X = df.review.to_list()
    y = df.label.to_list()
    categoricals = {"喜悦": 0, "愤怒": 1, "厌恶": 2, "低落": 3}
    return X, y, categoricals

_jobs = "/home/zhiwen/workspace/dataset/company-jobs/jobs.json"
def load_company_jobs(file=_jobs):
    filters = {'投融资', '移动开发', '高端技术职位', '行政', '运营', '人力资源',
               '后端开发', '市场/营销', '销售', '产品经理', '项目管理', '运维', '测试',
               '视觉设计', '编辑', '公关', '财务', '客服', '前端开发', '企业软件'}
    df = pd.read_json(file)
    df = df.sample(frac=1)
    X = []
    y = []
    for job in df["jobs"]:
        if job["type"] not in filters:
            continue
        X.append(job["desc"])
        y.append(job["type"])
    categoricals = list(set(y))
    categoricals.sort()
    categoricals = {label: i for i, label in enumerate(categoricals)}
    y = [categoricals[i] for i in y]
    return X, y, categoricals

_EMO = "/home/zhiwen/workspace/dataset/CLUEmotionAnalysis2020/CLUEdataset/emotion/"
def load_clue_emotion(file=_EMO):
    pass

class Tokenizer:
    """字转ID
    """

    def __init__(self, min_freq=16, cutword=False):
        self.char2id = {}
        self.MASK = 0
        self.UNKNOW = 1
        self.min_freq = min_freq
        self.cutword = cutword
        self.filters = set("!\"#$%&'()[]*+,-./，。！@·……（）【】<>《》?？；‘’“”")
        if cutword:
            import jieba
            jieba.initialize()
            self.lcut = jieba.lcut

    def fit(self, X):
        # 建立词ID映射表
        chars = collections.defaultdict(int)
        for c in itertools.chain(*X):
            chars[c] += 1

        if self.cutword:
            for x in X:
                for w in self.lcut(x):
                    chars[w] += 1

        # 过滤低频词
        chars = {i:j for i,j in chars.items() \
                 if j >= self.min_freq \
                 and i not in self.filters}
        # 0:MASK
        # 1:UNK
        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNKNOW))
            ids.append(s)

        if not self.cutword:
            return ids

        wids = []
        for sentence in X:
            w = []
            for word in self.lcut(sentence):
                # 字词对齐
                w += [self.char2id.get(word, self.UNKNOW)] * len(word)
            wids.append(w)
        return ids, wids

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.char2id) + 2

    @property
    def vocab(self):
        return self.char2id

def find_best_maxlen(X, mode="mean"):
    # 获取适合的截断长度
    ls = [len(sample) for sample in X]
    if mode == "mode":
        maxlen = np.argmax(np.bincount(ls))
    if mode == "mean":
        maxlen = np.mean(ls)
    if mode == "median":
        maxlen = np.median(ls)
    if mode == "max":
        maxlen = np.max(ls)
    return int(maxlen)

def find_embedding_dims(vocab_size):
    return np.ceil(8.5 * np.log(vocab_size)).astype("int")

if __name__ == "__main__":
    # for testing
    load_stop_words()
    load_THUCNews_title_label()
    load_taotiao_news()
    load_weibo_senti_100k()
    load_simplifyweibo_4_moods()
    load_company_jobs()
