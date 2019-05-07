# -*- coding: utf-8 -*-

import os
import numpy as np
import jieba_fast
from gensim.models import Word2Vec
import pickle
import time


#########################################################################
## sigmoid函数 ##
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#########################################################################
## 求X Y两个向量的cosine值
def get_cosine_value(X_list, Y_list, X_norm, Y_norm):
    # 分子 x1*y1 + x2*y2 + ... + xn*yn
    # 分母 X_norm * Y_norm

    if (X_norm <= 0.0 or Y_norm <= 0.0 or len(X_list) != len(Y_list)):
        return 0

    XY_size = len(X_list)

    X = X_list.reshape(1, XY_size)
    Y = Y_list.reshape(1, XY_size)

    return float(X.dot(Y.T) / (X_norm * Y_norm))


##################################一个聚类簇的信息#######################################
class SentenceEmbedding():
    def __init__(self):

        # 停用词文件路径
        self.stopword_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/all_stopword.txt")

        # 停用词集合
        self.stopword_list = []

        # 停用词加载
        self.load_stopword()

        # word2vec模型
        self.model = Word2Vec.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../word2vec/word2vec_wx"))

        # 意图文件路径
        self.intention_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/intention.txt")

        # idf文件路径
        self.idf_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/idf.txt")

        # 词idf dict
        self.word_idf_dict = {}

        # 没有的词的idf默认系数
        self.idf_default = 5.0

        # 加载所有词的idf 到 类变量中
        self.load_idf()

        # 所有句子的集合
        self.all_sentence_list = []



    # 加载所有的停用词 到 类变量中
    def load_stopword(self):

        for line in open(self.stopword_file):
            self.stopword_list.append(line.strip())


    # 加载所有词的idf 到 类变量中
    def load_idf(self):

        for line in open(self.idf_file):
            line_list = line.strip().split(" ")
            if 2 == len(line_list):
                self.word_idf_dict[line_list[0]] = float(line_list[1])


    # 获取某个词的word2vec值 256维
    def get_word2vec(self, word):

        if(word in self.model):
            return self.model[word]
        else:
            return np.zeros(0, dtype=np.float)


    # 获取某一个句子的句向量
    def get_sentence_embedding(self, sentence):

        # 句子的向量
        sentence_embedding = np.zeros(self.model.vector_size, dtype=np.float)

        rst = jieba_fast.cut(sentence)

        for word in rst:

            if word in self.stopword_list:
                # 停用词跳过
                continue

            # word2vec某词
            w2v_vector = self.get_word2vec(word)
            if 0 == len(w2v_vector):
                continue

            idf = self.word_idf_dict.get(word, self.idf_default)  # idf的默认系数

            # 可以开始计算了
            coe = sigmoid(idf)

            sentence_embedding += coe * w2v_vector
            #sentence_embedding = sentence_embedding + coe * w2v_vector

        sentence_norm = np.linalg.norm(sentence_embedding)

        return sentence_embedding, sentence_norm


    # 处理所有的预设的句子
    def proc_intention(self):

        start_num = 1

        for line in open(self.intention_file):
            line = line.strip()

            sentence_embedding, sentence_norm = self.get_sentence_embedding(line)

            sentence_info_dict = {}
            sentence_info_dict["sentence_id"]           = start_num
            sentence_info_dict["sentence"]              = line
            sentence_info_dict["sentence_embedding"]    = sentence_embedding
            sentence_info_dict["sentence_norm"]         = sentence_norm

            start_num += 1

            self.all_sentence_list.append(sentence_info_dict)





#########################################################################
## main 主函数 ##

if __name__ == '__main__':

    sentence_embedding = SentenceEmbedding()


    # 处理所有的预设的句子
    sentence_embedding.proc_intention()

    print("=================print_result=======================")
    print("sentence_embedding.all_sentence_list.len=%d" % len(sentence_embedding.all_sentence_list))
    #for i in range(len(sentence_embedding.all_sentence_list)):
    #    print(sentence_embedding.all_sentence_list[i])
    print("=================print_result=======================")

    #input_sentence = "股票名称智能诊股"
    input_sentence = "范式概念股票"

    input_sentence_embedding, input_sentence_norm = sentence_embedding.get_sentence_embedding(input_sentence)

    # 存放结果集合
    rst_list = []

    for i in range(len(sentence_embedding.all_sentence_list)):

        similar_value = get_cosine_value(input_sentence_embedding, \
                                         sentence_embedding.all_sentence_list[i]["sentence_embedding"], \
                                         input_sentence_norm, \
                                         sentence_embedding.all_sentence_list[i]["sentence_norm"])

        rst_list.append((sentence_embedding.all_sentence_list[i]["sentence"], similar_value))

    sort_rst_list = list.sort(rst_list, key=lambda rs: rs[1], reverse=True)
    print(rst_list)

