# -*- coding: utf-8 -*-

import os
import numpy as np
import jieba_fast
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD


# 用于计算论文A Simple but Tough-to-Beat Baseline for Sentence Embeddings
# 参考代码： https://github.com/PrincetonML/SIF
# 主要思路:
# 计算sentence的embedding，不再使用沙雕的相加的方式，而是带有权重的加法，使用怎么样的权重，请参考论文


##################################处理句子嵌入的类#######################################
class SentenceEmbedding():
    def __init__(self):

        print("SentenceEmbedding __init__")

        # 参数a的值  人工设定的
        self.para_A_float = 1.0

        # 停用词文件路径
        self.stopword_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/all_stopword.txt")
        print("stopword_file=%s" % self.stopword_file)

        # 停用词集合
        self.stopword_list = []

        # 停用词加载
        self.load_stopword()

        # word2vec模型
        self.model = Word2Vec.load("word2vec/word2vec_wx")

        # 意图文件路径
        self.intention_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/intention.txt")
        print("intention_file=%s" % self.intention_file)

        # 标题文件路径
        self.title_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/title.txt")
        print("title_file=%s" % self.title_file)

        # 词汇在样本语料中的频率(词汇出现次数/总词汇次数)
        self.word_proba_dict = {}

        # 记录所有语句转化成word_list的集合(注意这里是:集合的集合)
        self.word_list_list = []

        # 所有句子的向量集合 Vs
        self.all_sentence_embedding = np.empty([0, self.model.vector_size], dtype=np.float)

        # 投影的矩阵
        self.uuT = np.empty([0, 1], dtype=np.float)

        # 计算所有文本的所有单词出现的概率
        self.statistics_word_proba()


    # 加载所有的停用词 到 类变量中
    def load_stopword(self):

        for line in open(self.stopword_file):
            self.stopword_list.append(line.strip())



    # 获取某个词的word2vec值 256维
    def get_word2vec(self, word):

        if(word in self.model):
            return self.model[word]
        else:
            return np.zeros(0, dtype=np.float)


    # 第一步获取到的sentence embedding
    def get_first_sentence_embedding(self, word_list):

        # 句子的向量 Vs
        sentence_embedding = np.zeros(self.model.vector_size, dtype=np.float)

        # 有效词的个数 |s|
        word_num = 0

        for i in range(len(word_list)):
            word = word_list[i]
            if word not in self.word_proba_dict:
                continue

            # 某词出现的概率
            proba_word = self.word_proba_dict[word]

            # 系数
            coe = self.para_A_float/(self.para_A_float + proba_word)

            # word2vec某词
            w2v_vector = self.get_word2vec(word)
            if 0 == len(w2v_vector):
                continue

            sentence_embedding = sentence_embedding + coe * w2v_vector

            word_num = word_num + 1

        if(word_num > 0):
            sentence_embedding = sentence_embedding / word_num

        return sentence_embedding


    # 获取系数(1-u*u.T)
    def get_coe(self):

        # 获取所有的句子embedding
        # all_sentence_embedding is u
        all_sentence_embedding = np.empty([0, self.model.vector_size], dtype=np.float)

        # 遍历所有句子的word_list
        for i in range(len(self.word_list_list)):

            word_list = self.word_list_list[i]

            sentence_embedding = self.get_first_sentence_embedding(word_list)

            all_sentence_embedding = np.vstack((all_sentence_embedding, sentence_embedding))

        #保存所有向量的集合
        self.all_sentence_embedding = all_sentence_embedding

        # SVD or PCA
        uT_svd = TruncatedSVD(n_components=1).fit_transform(all_sentence_embedding)

        uuT = uT_svd.dot(uT_svd.T) / np.linalg.norm(uT_svd)

        self.all_sentence_embedding = self.all_sentence_embedding - uuT.dot(self.all_sentence_embedding)

        self.uuT = uuT



    # 计算所有文本的所有单词出现的概率
    # 去除停用词后 所有句子的单词累加/单词总数
    def statistics_word_proba(self):

        word_num_dict = {}
        sum_num = 0

        for line in open(self.intention_file):
        #for line in open(self.title_file):
            content_str = line.strip()

            rst = jieba_fast.cut(content_str)

            # 句子的词汇进行集合汇总
            word_list = []

            for word in rst:

                if word in self.stopword_list:
                    # 停用词跳过
                    continue

                if word in word_num_dict:
                    word_num_dict[word] = word_num_dict[word] + 1
                else:
                    word_num_dict[word] = 1

                sum_num = sum_num + 1

                word_list.append(word)

            # 汇总所有句子的word_list
            self.word_list_list.append(word_list)

        for word, num in word_num_dict.items():
            self.word_proba_dict[word] = num / sum_num


#########################################################################
## 求X Y两个向量的cosine值
def get_cosine_value(X, Y):
    # 分子 x1*y1 + x2*y2 + ... + xn*yn
    # 分母 ||X|| * ||Y||

	if (np.linalg.norm(X) <= 0.0 or np.linalg.norm(Y) <= 0.0):
        return 0

    X = X.reshape(1, 256)
    Y = Y.reshape(1, 256)

    return float(X.dot(Y.T) / (np.linalg.norm(X) * np.linalg.norm(Y)))


#########################################################################
## main 主函数

if __name__ == '__main__':

    sentence_embedding_class = SentenceEmbedding()

    # print(sentence_embedding_class.word_proba_dict)
    # print("----------------get_coe--------------------")
    sentence_embedding_class.get_coe()

    #####################################################################################
    #content_str = "四幅图显示：黄金正在触底 重大行情一触即发"
    content_str = "股票名称智能诊股"

    rst = jieba_fast.cut(content_str)

    # 句子的词汇进行集合汇总
    word_list = []

    for word in rst:

        if word in sentence_embedding_class.stopword_list:
            # 停用词跳过
            continue

        word_list.append(word)

    sentence_embedding = sentence_embedding_class.get_first_sentence_embedding(word_list)

    #input_sentence_embedding = sentence_embedding - sentence_embedding_class.uuT.dot(sentence_embedding)

    input_sentence_embedding = sentence_embedding_class.all_sentence_embedding[0]

    #####################################################################################

    # print("-------------------------print--------------------------")
    # print(np.shape(sentence_embedding_class.all_sentence_embedding))
    # print(len(sentence_embedding_class.all_sentence_embedding))
    # print(sentence_embedding_class.all_sentence_embedding[0])

    # 存放结果集合
    rst_list = []

    for i in range(len(sentence_embedding_class.all_sentence_embedding)):
        sample_sentenc_embedding = sentence_embedding_class.all_sentence_embedding[i]

        conine_rst_float = get_cosine_value(input_sentence_embedding, sample_sentenc_embedding)

        rst_list.append((i+1, conine_rst_float))

    #print(rst_list)
    sort_rst_list = list.sort(rst_list, key=lambda rs:rs[1], reverse=True)
    print(rst_list)


