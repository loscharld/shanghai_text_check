#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import re
import jieba
import jieba.analyse
import numpy as np
import json
import pandas as pd
import datetime

from gensim import corpora,models,similarities  #用于tf-idf

# from gensim.models.word2vec import LineSentence, Word2Vec

from collections import defaultdict
# import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

#连数据库
import cx_Oracle
from sqlalchemy import create_engine
# %matplotlib inline
from sklearn.ensemble import IsolationForest
import warnings

def getData(user,password,database,targetTable,commandText):
    connection = cx_Oracle.connect(user,password,database)
    cursor = connection.cursor()
    cursor.execute(commandText.format(targetTable))
    x = cursor.description
    columns = [y[0] for y in x]
    cursor01 = cursor.fetchall()
    cursor.close()
    data = pd.DataFrame(cursor01,columns = columns)
    return data


# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

#将数据上传到库中
import sys
sys.path.append(r"E:\dingli\修理厂数据上传")

from ForCall01 import useOracle

# python中对NLS_LANG实现设置
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

user = 'lbcc'
password = 'xdf123'
database = 'orcl'
targetTable = 'soure_01'
commandText = '''select t.id,t.word from SOURE_02 t where rownum <= 10
            '''
txt_data = getData(user,password,database,targetTable,commandText)
print(txt_data.shape)
print(txt_data.columns)
txt_data.head(5)


# 将文章每十句合成一个段落
def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def chaifen_doc(doc):
    punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:《》'

    #     with open(doc,encoding='utf-8') as f:
    #         data = f.read()
    list1 = re.split(r"([。!！?？])", doc)
    list1.append("")
    z_list1 = ["".join(i) for i in zip(list1[0::2], list1[1::2])]  # 将文章分成每一句为一个元素存放于列表中

    exam = func(z_list1, 10)

    list_1 = []
    list_2 = []
    for i in exam:
        t1 = "".join(i)
        t2 = re.sub(r'[^\w]+', '', t1)
        X, Y = ['\u4e00', '\u9fa5']
        s = jieba.cut(t2)
        s = [i for i in s if len(i) > 1 and X <= i <= Y]  # 匹配汉字且不在停用词内的
        string = ''
        string = string.join(s)  # 合并分词
        line1 = re.sub(r"[{}]+".format(punc), "", string)
        list_1.append(t1)
        list_2.append(line1)
    return z_list1, list_1, list_2


def ju_chaifen(ju):
    list1 = re.split(r"([。!！?？])", ju)
    list1.append("")
    list1 = ["".join(i) for i in zip(list1[0::2], list1[1::2])]
    list_11 = []
    for i in list1:
        punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:《》'

        t1 = re.sub(r'[^\w]+', '', i)
        X, Y = ['\u4e00', '\u9fa5']
        s = jieba.cut(t1)
        s = [i for i in s if len(i) > 1 and X <= i <= Y]  # 匹配汉字且不在停用词内的
        string = ''
        string = string.join(s)  # 合并分词
        line1 = re.sub(r"[{}]+".format(punc), "", string)
        list_11.append(line1)
    return list_11


def ju_sim(duibi_doc, doc):
    documents = []
    for i in doc:
        data1 = jieba.cut(i)
        data11 = ''
        for i in data1:
            data11 += i + ' '
        documents.append(data11)

    texts = [[word for word in document.split()] for document in documents]

    frequency = defaultdict(int)
    for text in texts:
        for word in text:
            frequency[word] += 1

    # 5、对频率低的词语进行过滤（可选）
    # texts=[[word for word in text if frequency[word]>6] for text in texts]    #频率低的界限设定

    # 6、通过语料库将文档的词语进行建立词典
    dictionary = corpora.Dictionary(texts)

    j_jieguo = pd.DataFrame()
    sum1 = 0
    for i in duibi_doc:
        data3 = jieba.cut(i)
        data31 = ""
        for i in data3:
            data31 += i + " "
        new_xs = dictionary.doc2bow(data31.split())
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        featurenum = len(dictionary.token2id.keys())
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=featurenum)

        sim = index[tfidf[new_xs]].tolist()

        if max(sim) > 0.7:
            sum1 += 1
    if len(duibi_doc[-1]) == 0:
        ju_sim = sum1 / (len(duibi_doc) - 1)
    else:
        ju_sim = sum1 / (len(duibi_doc))
    return ju_sim


a = datetime.datetime.now()
n = 0
aaa = []
textsall = []
list11 = []
for i in range(len(txt_data)):
    doc_1 = str(txt_data['WORD'][i])

    z_1, list_1, list_2 = chaifen_doc(doc_1)
    z_1_ = [i for i in z_1 if len(i) > 10]  # 每句话少于10的不计入
    list11.append(list_1)
    # 段落比对
    documents = []
    for i in list_2:
        data1 = jieba.cut(i)
        data11 = ''
        for i in data1:
            data11 += i + ' '
        documents.append(data11)

    texts = [[word for word in document.split()] for document in documents]
    textsall.append(texts)
    frequency = defaultdict(int)
    for text in texts:
        for word in text:
            frequency[word] += 1
    aaa.append(frequency)
txt_data['frequency'] = aaa
txt_data['texts'] = textsall
txt_data['list11'] = list11
b = datetime.datetime.now()
print('所用时间为：' + str((b - a).seconds) + "秒")


a1 = datetime.datetime.now()
texts11 = txt_data['texts'].tolist()
list11 = txt_data['list11'].tolist()
indexall = []
tfidfall = []
for i in range(len(txt_data)):
    dictionary=corpora.Dictionary(texts11[i])
    corpus = [dictionary.doc2bow(text)for text in texts11[i]]
    #10、将新语料库通过tf-idf model 进行处理，得到tfidf
    tfidf = models.TfidfModel(corpus)
    # 11、通过token2id得到特征数
    featurenum = len(dictionary.token2id.keys())
    # 12、稀疏矩阵相似度，从而建立索引
    index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=featurenum)
    indexall.append(index)
    tfidfall.append(tfidf)
b1 = datetime.datetime.now()
print('所用时间为：'+ str((b1-a1).seconds)+"秒")

doc = r'txt4/234.txt'     #需要比对的文章

with open(doc,encoding='utf-8') as f:
    data = f.read()
    z_2,list_3,list_4 = chaifen_doc(data)

    e = datetime.datetime.now()
    texts11 = txt_data['texts'].tolist()
    list11 = txt_data['list11'].tolist()
    for i in range(len(txt_data)):
        dictionary = corpora.Dictionary(texts11[i])
        doc_name = str(txt_data['ID'][i])
        jieguo = pd.DataFrame()
        doc_index = {}
        count = 0
        for j in list_4:
            data3 = jieba.cut(j)
            data31 = ""
            for s in data3:
                data31 += s + " "
            # 8、将要对比的文档通过doc2bow转化为稀疏向量
            new_xs = dictionary.doc2bow(data31.split())
            #         print(new_xs)
            #         # 9、对语料库进一步处理，得到新语料库
            #         corpus = [dictionary.doc2bow(text)for text in texts11[i]]
            #         #10、将新语料库通过tf-idf model 进行处理，得到tfidf
            #         tfidf = models.TfidfModel(corpus)
            #         # 11、通过token2id得到特征数
            #         featurenum = len(dictionary.token2id.keys())
            #         # 12、稀疏矩阵相似度，从而建立索引
            #         index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=featurenum)
            #         for i1 in enumerate(index):
            #             print(i1)
            #       # 13、得到最终相似结果
            sim = indexall[i][tfidfall[i][new_xs]].tolist()
            #         print(sim)
            sim1 = pd.DataFrame(sim)
            d_jieguo = pd.concat([jieguo, sim1], axis=1)  # 将段落比对的结果放在数据框中
            d_index = []
            for j1 in sim:  # 分析段落比对的结果，将满足条件的段落分析句子相似
                if j1 >= 0.4:
                    index1 = sim.index(j1)
                    #                 print(index1)
                    d_index.append(index1)
            doc_index[str(count)] = d_index  # 得到判断后的相似与否的字典
            count += 1
        # 句子相似度
        sim = []
        for key, value in doc_index.items():
            #         print(key,value)
            if len(value) > 0:
                xuhao = int(key)
                name1 = '第' + str(xuhao + 1) + '段'
                for m in value:
                    xuhao_1 = m + 1
                    name2 = '与对比文' + str(xuhao_1) + '段'
                    e_1 = ju_chaifen(list_3[xuhao])
                    #                 print(list12[m])
                    e_2 = ju_chaifen(list11[i][m])
                    s = ju_sim(e_1, e_2)
                    #                         print(list_3[xuhao])
                    #                         print(list_1[i])

                    print(name1+name2+'相似度：'+str(s))

                    if len(e_1[-1]) == 0:
                        s_1 = s * (len(e_1) - 1)
                    else:
                        s_1 = s * (len(e_1))
                    sim.append(s_1)

        zong_sim = sum(sim) / len(z_2)
        if zong_sim > 0:
            #         print('与'+str(filename)+'相似度为：',zong_sim)
            print('与' + doc_name + '相似度为：', zong_sim)
    f = datetime.datetime.now()
    print('所用时间为：' + str((f - e).seconds) + "秒")