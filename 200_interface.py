import codecs
from gensim.models.doc2vec import Doc2Vec,LabeledSentence,TaggedDocument
from jieba import analyse
import jieba.posseg as pesg
import cx_Oracle
import re
import jieba
import jieba.analyse
import numpy as np
import json
import pandas as pd
import networkx as nx
import jieba.posseg as pseg
from matplotlib import pyplot as plt
import datetime
from gensim import corpora,models,similarities  #用于tf-idf
# from gensim.models.word2vec import LineSentence, Word2Vec
from collections import defaultdict
# import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from socketserver import ThreadingMixIn
class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass
import os
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'
# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK'

class GraphPlot():

    def __init__(self):
        self.fig = plt.figure()
        plt.ion()  # 允许更新画布
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.rcParams['figure.figsize'] = (18.0, 14.0)  # 设置figure_size尺寸

    def graph_plot(self, link_weight_list):
        plt.cla()  # 清除画布上原来的图像
        G = nx.Graph()
        for node1, node2, weight in link_weight_list:  # 添加节点以及边的权重
            G.add_edge(str(node1), str(node2), weight=weight)

        weights = [d['weight'] for u, v, d in G.edges(data=True)]
        pos = nx.circular_layout(G)  # 借点我围成一个圈，便于展示边的情况
        nx.draw_networkx_edges(G, pos, font_size=14, width=weights)  # 绘制图中边的权重
        nx.draw_networkx(G, pos, node_size=400)
        plt.pause(2)  # 展示出来定格2秒

# 用于存储图
class Graph():
    def __init__(self):
        self.linked_node_map = {}  # 邻接表，
        self.PR_map = {}  # 存储每个节点的入度
        self.stop_words = set({'我'})

    def clean(self):
        self.linked_node_map = {}  # 邻接表，
        self.PR_map = {}  # 存储每个节点的入度

    # 添加节点
    def add_node(self, node_id):
        if node_id not in self.linked_node_map:
            self.linked_node_map[node_id] = set({})
            self.PR_map[node_id] = 0
        else:
            print("这个节点已经存在")

    # 增加一个从Node1指向node2的边。允许添加新节点
    def add_link(self, node1, node2):
        if node1 not in self.linked_node_map:
            self.add_node(node1)
        if node2 not in self.linked_node_map:
            self.add_node(node2)
        self.linked_node_map[node1].add(node2)  # 为node1添加一个邻接节点，表示ndoe2引用了node1

    # 计算pr
    def get_PR(self, epoch_num=5, d=0.8, if_show=False):  # 配置迭代轮数，以及阻尼系数
        if if_show:
            GP = GraphPlot()  # 用于plot网络的实例
        for i in range(epoch_num):
            for node in self.PR_map:
                # 遍历每一个节点
                self.PR_map[node] = (1 - d) + d * sum(
                    [self.PR_map[temp_node] for temp_node in self.linked_node_map[node]])  # 原始版公式
                # print(self.PR_map)

            # 展示节点之间的链接权重
            if if_show:
                link_weight_list = []
                topN_nodes = set(list(map(lambda x: x[0], self.get_topN(10))))
                for source, targets in self.linked_node_map.items():
                    if source not in topN_nodes: continue
                    for t in targets:
                        link_weight_list.append([source, t, self.PR_map[source]])
                GP.graph_plot(link_weight_list)
            # 展示节点之间的链接权重

    def get_topN(self, top_n):
        topN = sorted(self.PR_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return topN

class TextRankSummary(Graph):

    # 切分句子。simple模式：基于句号分割
    def text2words(self, text):
        sentences = text.replace('\n', '').replace('\r','').replace(' ', '').split("。")
        sentences = list(filter(lambda x: len(x) > 1, sentences))
        words_list = list(map(lambda x: self.word_segment(x), sentences))
        new_sentences, new_words_list = [], []
        for i in range(len(sentences)):
            if len(words_list[i]) > 0:
                new_sentences.append(sentences[i])
                new_words_list.append(words_list[i])
        return new_sentences, new_words_list

    def word_segment(self, sentence):
        word_tag_list = pseg.cut(sentence)
        words = []
        for word_tag in word_tag_list:
            word_tag = str(word_tag).split('/')
            if len(word_tag) == 2:
                word, tag = word_tag
                if 'n' == tag and word not in self.stop_words:
                    words.append(word)
        return set(words)

    # 基于字，计算杰卡德相似度
    def sentence_simlarity(self, words1, words2):
        word_set1, word_set2 = set(words1), set(words2)
        simlarity = len(word_set1 & word_set2) / len(word_set2 | word_set2)
        return word_set1, word_set2,simlarity

    def get_sentence_links(self, sentences):

        sentence_link_list = []
        for s_id in range(len(sentences)):
            for s_jd in range(s_id+1, len(sentences)):
                word_set1, word_set2, simlarity=self.sentence_simlarity(sentences[s_id], sentences[s_jd])
                if simlarity > 0.5:
                    sentence_link_list.append([s_id, s_jd])
        return sentence_link_list

    def get_summary_with_textrank(self, text):
        self.clean()
        sentences, words_list = self.text2words(text)
        sentence_link_list = self.get_sentence_links(words_list)
        for link in sentence_link_list:
            self.add_link(link[0], link[1])
        self.get_PR()
        result = self.get_topN(5)
        summary_sentences = [sentences[i] for i, _ in result]
        return summary_sentences

def getData(user, password, database,commandText):
    connection = cx_Oracle.connect(user, password, database)
    cursor = connection.cursor()
    cursor.execute(commandText)
    x = cursor.description
    columns = [y[0] for y in x]
    cursor01 = cursor.fetchall()
    cursor.close()
    data = pd.DataFrame(cursor01, columns=columns)
    return data

def Comparison_two(fr1, fr2):
    punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:'
    stoplist = {}.fromkeys([line.rstrip() for line in
                            codecs.open(r"中文停用词库.txt", 'r', 'utf-8')])  # 建立字典，词库中的字符为键
    # with open(fr1, 'r', encoding='utf-8') as f:
    #     list1 = f.read()
    string = ''
    X, Y = ['\u4e00', '\u9fa5']
    text1 = re.sub(r'[^\w]+', '', fr1)  # 将文本中的特殊字符去掉
    # print(text1)
    s = jieba.cut(text1)
    s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]  # 匹配汉字且不在停用词内的
    string = string.join(s)  # 合并分词
    line1 = re.sub(r"[{}]+".format(punc), "", string)  # 将文本中标点符号去掉
    # word_list1 = [word.word for word in pesg.cut(line1) if word.flag[0] not in ['w', 'x', 'u']]
    word_list1=extract1(line1)
    # with open(fr2, 'r', encoding='utf-8') as f:
    #     list2 = f.read()
    string = ''
    X, Y = ['\u4e00', '\u9fa5']
    text2 = re.sub(r'[^\w]+', '', fr2)
    # print(text2)
    s = jieba.cut(text2)
    s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]
    string = string.join(s)
    line2 = re.sub(r"[{}]+".format(punc), "", string)
    # word_list2 = [word.word for word in pesg.cut(line2) if word.flag[0] not in ['w', 'x', 'u']]
    word_list2=extract1(line2)
    return word_list1, word_list2

def Comparison_two2(fr1, fr2):
    punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:'
    stoplist = {}.fromkeys([line.rstrip() for line in
                            codecs.open(r"中文停用词库.txt", 'r', 'utf-8')])  # 建立字典，词库中的字符为键
    # with open(fr1, 'r', encoding='utf-8') as f:
    #     list1 = f.read()
    string = ''
    X, Y = ['\u4e00', '\u9fa5']
    text1 = re.sub(r'[^\w]+', '', fr1)  # 将文本中的特殊字符去掉
    # print(text1)
    s = jieba.cut(text1)
    s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]  # 匹配汉字且不在停用词内的
    string = string.join(s)  # 合并分词
    line1 = re.sub(r"[{}]+".format(punc), "", string)  # 将文本中标点符号去掉
    # word_list1 = [word.word for word in pesg.cut(line1) if word.flag[0] not in ['w', 'x', 'u']]
    word_list1=extract2(line1)
    # with open(fr2, 'r', encoding='utf-8') as f:
    #     list2 = f.read()
    string = ''
    X, Y = ['\u4e00', '\u9fa5']
    text2 = re.sub(r'[^\w]+', '', fr2)
    # print(text2)
    s = jieba.cut(text2)
    s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]
    string = string.join(s)
    line2 = re.sub(r"[{}]+".format(punc), "", string)
    # word_list2 = [word.word for word in pesg.cut(line2) if word.flag[0] not in ['w', 'x', 'u']]
    word_list2=extract2(line2)
    return word_list1, word_list2

def sent2vec(model, words):
    """文本转换成向量

    Arguments:
        model {[type]} -- Doc2Vec 模型
        words {[type]} -- 分词后的文本

    Returns:
        [type] -- 向量数组
    """

    vect_list = []
    for w in words:
        try:
            vect_list.append(model.wv[w])
        except:
            continue
    vect_list = np.array(vect_list)
    vect = vect_list.sum(axis=0)
    return vect / np.sqrt((vect ** 2).sum())

def ceshi(text_text):
    model_dm=Doc2Vec.load('model_doc/model_sim')
    inferred_vector_dm=model_dm.infer_vector(text_text.split(' '))
    # print(len(inferred_vector_dm))
    sims=model_dm.docvecs.most_similar([inferred_vector_dm],topn=10)
    return sims

def extract(line):
    #引用TF_IDF关键词抽取接口
    tfidf=analyse.extract_tags
    str1_fenci=' '.join(jieba.cut(line))
    # stop_word=[]
    # with open('中文停用词库.txt','r',encoding='utf-8') as fp:
    #     for line in fp.readlines():
    #         line=line.strip()
    #         if line=='':
    #             continue
    #         stop_word.append(line)
    str1_rv_stop_word=''
    str1_rv_stop_word_fenci=''
    for each in list(set(str1_fenci.split(' '))):
        # if each not in stop_word:
        if str1_rv_stop_word=='':
            str1_rv_stop_word=each
            str1_rv_stop_word_fenci=each
        else:
            str1_rv_stop_word=str1_rv_stop_word+each
            str1_rv_stop_word_fenci=str1_rv_stop_word_fenci+' '+each

    guanjian=tfidf(str1_rv_stop_word_fenci,topK=910)
    guanjian_result=''
    lianshi=[]
    for each in str1_rv_stop_word_fenci.split(' '):
        if each in guanjian:
            if guanjian_result=='':
                guanjian_result=each
            else:
                guanjian_result=guanjian_result+' '+each

            lianshi.append(each)

    return guanjian_result


def word_segment(sentence):
    sentence=sentence.replace(' ','')
    word_tag_list = pseg.cut(sentence)
    words = []
    for word_tag in word_tag_list:
        word_tag = str(word_tag).split('/')
        if len(word_tag) == 2:
            word, tag = word_tag
            if 'n' == tag:
                words.append(word)
    return list(set(words))


def extract1(line):
    #引用TF_IDF关键词抽取接口
    tfidf=analyse.extract_tags
    str1_fenci=' '.join(jieba.cut(line))
    # stop_word=[]
    # with open('中文停用词库.txt','r',encoding='utf-8') as fp:
    #     for line in fp.readlines():
    #         line=line.strip()
    #         if line=='':
    #             continue
    #         stop_word.append(line)
    str1_rv_stop_word=''
    str1_rv_stop_word_fenci=''
    for each in list(set(str1_fenci.split(' '))):
        # if each not in stop_word:
        if str1_rv_stop_word=='':
            str1_rv_stop_word=each
            str1_rv_stop_word_fenci=each
        else:
            str1_rv_stop_word=str1_rv_stop_word+each
            str1_rv_stop_word_fenci=str1_rv_stop_word_fenci+' '+each

    guanjian=tfidf(str1_rv_stop_word_fenci,topK=100)
    guanjian_result=''
    lianshi=[]
    for each in str1_rv_stop_word_fenci.split(' '):
        if each in guanjian:
            if guanjian_result=='':
                guanjian_result=each
            else:
                guanjian_result=guanjian_result+' '+each

            lianshi.append(each)
    #提取名词关键词
    guanjian_result=word_segment(guanjian_result)
    return guanjian_result

def extract2(line):
    # 引用TF_IDF关键词抽取接口
    tfidf = analyse.extract_tags
    str1_fenci = ' '.join(jieba.cut(line))
    # stop_word=[]
    # with open('中文停用词库.txt','r',encoding='utf-8') as fp:
    #     for line in fp.readlines():
    #         line=line.strip()
    #         if line=='':
    #             continue
    #         stop_word.append(line)
    str1_rv_stop_word = ''
    str1_rv_stop_word_fenci = ''
    for each in list(set(str1_fenci.split(' '))):
        # if each not in stop_word:
        if str1_rv_stop_word == '':
            str1_rv_stop_word = each
            str1_rv_stop_word_fenci = each
        else:
            str1_rv_stop_word = str1_rv_stop_word + each
            str1_rv_stop_word_fenci = str1_rv_stop_word_fenci + ' ' + each

    guanjian = tfidf(str1_rv_stop_word_fenci, topK=200)
    guanjian_result = ''
    lianshi = []
    for each in str1_rv_stop_word_fenci.split(' '):
        if each in guanjian:
            if guanjian_result == '':
                guanjian_result = each
            else:
                guanjian_result = guanjian_result + ' ' + each

            lianshi.append(each)

    # 提取名词关键词
    guanjian_result = word_segment(guanjian_result)
    return guanjian_result

def result_sim(doc):

    ceshi_list = []
    punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:'
    stoplist = {}.fromkeys([line.rstrip() for line in
                            codecs.open(r"中文停用词库.txt", 'r', 'utf-8')])
    # with open(path, 'r', encoding='utf-8') as f:
    # list1 = f.read()
    string = ''
    X, Y = ['\u4e00', '\u9fa5']
    text1 = re.sub(r'[^\w]+', '', doc)  # 将文本中的特殊字符去掉
    # print(text1)
    s = jieba.cut(text1)
    s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]  # 匹配汉字且不在停用词内的
    string = string.join(s)  # 合并分词
    line1 = re.sub(r"[{}]+".format(punc), "", string)  # 将文本中标点符号去掉
    ceshi_list.append(line1)
    for line in ceshi_list:
        # f1.write(line + '\n' + '\n')
        line = extract(line)
        sims = ceshi(line)
        num=0
        # df1 = pd.DataFrame()
        doc2sim_all={}
        doc2sim_all_1=[]
        for count, sim in sims:
            doc2sim = {}
            num+=1
            with open('data/index2id.json') as f:
                index2id = json.load(f)
            id=index2id[str(count)]
            user = 'lbcc'
            password = 'xdf123'
            database = 'orcl'
            targetTable = 'soure_01'
            commandText = '''select t.word,t.file_path from SOURE_02 t  where t.id={}
                                                '''.format(id)
            document=getData(user,password,database,targetTable,commandText)
            content=str(document['WORD'][0])
            file=str(document['FILE_PATH'][0])
            sim=round(sim,2)
            doc2sim['content']=content
            doc2sim['sim_score']=sim
            doc2sim['Fname'] =file
            doc2sim_all_1.append(doc2sim)
        doc2sim_all['data']=doc2sim_all_1
    return  doc2sim_all

def similarity(vector1 , vector2):
    """计算两个向量余弦值

    Arguments:
        a_vect {[type]} -- a 向量
        b_vect {[type]} -- b 向量

    Returns:
        [type] -- [description]
    """
    cos1 = np.sum(vector1 * vector2)
    cos21 = np.sqrt(sum(vector1 ** 2))
    cos22 = np.sqrt(sum(vector2 ** 2))
    similarity = cos1 / float(cos21 * cos22)
    return similarity

def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def chaifen_doc(doc):
    punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:《》'

    #     with open(doc,encoding='utf-8') as f:
    #         data = f.read()
    list1 = re.split(r"([。!！])", doc)
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
    list1 = re.split(r"([。!！])", ju)
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


def wb1_sim(duibi_doc,doc,n1=0.4):   #n1:段落相似的设定
#     duibi_doc = json.loads(duibi_doc)
#     doc = json.loads(doc)
    db_z_2,db_3,db_4 = chaifen_doc(duibi_doc)
    db_z_2 = [i for i in db_z_2 if len(i)>10]
    txt_z_1,txt_1,txt_2 = chaifen_doc(doc)
    #         z_2,list_3,list_4 = chaifen_doc(bidui_doc)    #需比对的文档
    txt_z_1 = [i for i in txt_z_1 if len(i)>10]      #每句话少于10的不计入
    #         z_2 = [i for i in z_2 if len(i)>10]

        #段落比对
    documents = []
    for i in txt_2:
        data1 = jieba.cut(i)
        data11 = ''
        for i in data1:
            data11 += i+' '
        documents.append(data11)

    texts = [[word for word in document.split()] for document in documents]

    frequency = defaultdict(int)
    for text in texts:
        for word in text:
            frequency[word] += 1

    #5、对频率低的词语进行过滤（可选）
    # texts=[[word for word in text if frequency[word]>6] for text in texts]    #频率低的界限设定

    # 6、通过语料库将文档的词语进行建立词典
    dictionary=corpora.Dictionary(texts)
    # d3 = du_doc(doc3)
    jieguo = pd.DataFrame()
    doc_index = {}
    count = 0
    for i in db_4:
        data3 = jieba.cut(i)
        data31 = ""
        for i in data3:
            data31 += i+" "
        #8、将要对比的文档通过doc2bow转化为稀疏向量
        new_xs = dictionary.doc2bow(data31.split())
        #9、对语料库进一步处理，得到新语料库
        corpus = [dictionary.doc2bow(text)for text in texts]
        #10、将新语料库通过tf-idf model 进行处理，得到tfidf
        tfidf = models.TfidfModel(corpus)
        #11、通过token2id得到特征数
        featurenum = len(dictionary.token2id.keys())
        #12、稀疏矩阵相似度，从而建立索引
        index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=featurenum)
        #13、得到最终相似结果
        sim = index[tfidf[new_xs]].tolist()
        sim1 = pd.DataFrame(sim)
        d_jieguo = pd.concat([jieguo,sim1],axis=1)     #将段落比对的结果放在数据框中
        d_index = []
        for j in sim:      #分析段落比对的结果，将满足条件的段落分析句子相似
            if j >= n1:
                index1 = sim.index(j)
    #                 print(index1)
                d_index.append(index1)
        doc_index[str(count)] = d_index    #得到判断后的相似与否的字典
        count += 1

    #句子相似度


    sim = []
    doc_sim_all = {}
    doc_sim_all_1 = []
    for key,value in doc_index.items():
#         print(key,value)
        if len(value) >0:
            xuhao = int(key)
            name1 = '第'+str(xuhao+1)+'段'

            doc_sim = {}
            for i in value:
                xuhao_1 = i+1
                name2 = '与对比文'+str(xuhao_1)+'段'
                e_1 = ju_chaifen(db_3[xuhao])
#                 print(txt_1[i])
                e_2 = ju_chaifen(txt_1[i])
                s = ju_sim(e_1,e_2)
                doc_sim[name1+name2+'相似度：'] = str(s)
                doc_sim_all_1.append(doc_sim)

#                 print(name1+name2+'相似度：'+str(s))

                if  len(e_1[-1]) == 0:
                    s_1 =  s*(len(e_1)-1)
                else:
                    s_1 = s*(len(e_1))
                sim.append(s_1)

    zong_sim = sum(sim)/len(db_z_2)
    doc_sim_all['compare'] = doc_sim_all_1
    # doc_sim_all['tscore']=str(zong_sim)
    return doc_sim_all



class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


# Create server
# with SimpleXMLRPCServer(("127.0.0.1", 5002),requestHandler=RequestHandler) as server:
#     server.register_introspection_functions()

# Create server
with ThreadXMLRPCServer(("127.0.0.1", 5004), requestHandler=RequestHandler) as server:
    server.register_introspection_functions()
    def wb_sim(duibi_doc, doc, n1=0.5):
        print("load model")
        doc_sim_all={}
        doc_sim={}
        model = Doc2Vec.load('model_doc/model_sim')
        st1, st2 =Comparison_two(duibi_doc, doc)
        # 转成句子向量
        vect1 = sent2vec(model, st1)
        vect2 = sent2vec(model, st2)
        cos = similarity(vect1, vect2)
        two_sim=round(cos, 3)
        if float(two_sim) < n1:
            print('两篇文章不相似！')
            # print('两篇文章相似度为：{}'.format(round(two_sim,3)))
        else:
            print('两篇文章相似！')
            # print('两篇文章相似度为：｛｝'.format(round(two_sim, 3)))
        # print("相似度：{:.4f}".format(cos))
        doc_n = 'sim_score'
        doc_sim['Stxt']=duibi_doc
        doc_sim['Ctxt']=doc
        doc_sim[doc_n] = str(two_sim)
        doc_sim_all['data']=doc_sim
        return doc_sim_all

    def multiple_sim(files1,files2,n1):
        print("load model")
        # Dir=os.listdir(files)
        # Dir=[files+'/'+i for i in Dir]
        user = 'lbcc'
        password = 'xdf123'
        database = 'orcl'

        pathes1=files1
        pathes2=files2
        # if len(pathes)<2:
        #     return '输入的文档数目必须是两篇及两篇以上'
        doc_sim_all =pd.DataFrame()
        model = Doc2Vec.load('model_doc/model_sim')
        def extract_content():
            top_nr = ''
            top_cxd = ''
            top_jsgj = ''
            try:
                for line in top.split('\n'):
                    if '研究内容' in line:
                        content = line.strip()
                        if len(content) < 25:
                            top_nr += top.split(line)[1][:1000]
            except:
                pass

            try:
                for line in top.split('\n'):
                    if '创新点' in line:
                        content = line.strip()
                        if len(content) < 15:
                            top_cxd += top.split(line)[1][:300]
            except:
                pass

            try:
                for line in top.split('\n'):
                    if '技术关键' in line:
                        content = line.strip()
                        if len(content) < 12:
                            top_jsgj += top.split(line)[1][:600]
            except:
                pass

            tail_nr = ''
            tail_cxd = ''
            tail_jsgj = ''
            try:
                for line in tail.split('\n'):
                    if '研究内容' in line:
                        content = line.strip()
                        if len(content) < 25:
                            tail_nr += tail.split(line)[1][:1000]
            except:
                pass

            try:
                for line in tail.split('\n'):
                    if '创新点' in line:
                        content = line.strip()
                        if len(content) < 15:
                            tail_cxd += tail.split(line)[1][:300]
            except:
                pass

            try:
                for line in tail.split('\n'):
                    if '技术关键' in line:
                        content = line.strip()
                        if len(content) < 12:
                            tail_jsgj += tail.split(line)[1][:600]
            except:
                pass

            top1 = top_nr + top_cxd + top_jsgj
            tail1 = tail_nr + tail_cxd + tail_jsgj
            return top1,tail1


        for i in range(len(pathes1)):
            print(i)
            path1=pathes1[i].replace('/','\\')
            commandText1 = '''select t.word from SOURE_02 t where t.FILE_PATH='{}' '''.format(path1)
            if pathes1[i].endswith('.json'):
                top=json.load(open(path1))
            else:
                top=str(getData(user,password,database,commandText1)['WORD'][0])

            try:
                if '中文摘要' in top:
                    top_zy=top.split('中文摘要')[-1][14:414]
                else:
                    top_zy =''
            except:
                pass



            path2 = pathes2[i].replace('/', '\\')
            commandText2 = '''select t.word from SOURE_02 t where t.file_path='{}' '''.format(path2)
            if pathes2[i].endswith('.json'):
                tail = json.load(open(path1))
            else:
                tail = str(getData(user, password, database, commandText2)['WORD'][0])

            try:
                if '中文摘要' in tail:
                    tail_zy = tail.split('中文摘要')[-1][14:414]
                else:
                    tail_zy=''
            except:
                pass
            #提取共同的关键词
            def extract_tg_wd(words):
                count_result = pd.Series(words).value_counts()
                count_dict = count_result.to_dict()
                re = [i for i in count_dict if count_dict[i] > 1]
                return re
            left_keywords=[]
            left_keywords_len=[]
            right_keywords=[]
            right_keywords_len=[]
            together_keywords=[]
            together_keywords_len=[]
            if top_zy and tail_zy:
                try:
                    st1, st2 = Comparison_two(top_zy, tail_zy)
                    zy_together_keywords=extract_tg_wd(st1+st2)
                    # 转成句子向量
                    vect1 = sent2vec(model, st1)
                    vect2 = sent2vec(model, st2)
                    cos = similarity(vect1, vect2)
                    cos -= 0.1
                    two_sim1 = round(cos, 3)
                    if two_sim1>=0.85:
                        sim_result=two_sim1
                        left_keywords=st1
                        len_left=len(left_keywords)
                        left_keywords_len.append(len_left)
                        right_keywords=st2
                        len_right=len(right_keywords)
                        right_keywords_len.append(len_right)
                        together_keywords=zy_together_keywords
                        len_together=len(together_keywords)
                        together_keywords_len.append(len_together)

                    else:
                        try:
                            top1,tail1=extract_content()
                            st9, st10 = Comparison_two2(top1, tail1)
                            nr_together_keywords=extract_tg_wd(st9+st10)
                            # 转成句子向量
                            try:
                                vect1 = sent2vec(model, st9)
                                vect2 = sent2vec(model, st10)
                                cos = similarity(vect1, vect2)
                            except:
                                cos=0.3
                            if len(top1)>900 and len(tail1)>900:
                                cos -= 0.12
                            else:
                                cos -= 0.15
                            two_sim2=round(cos, 3)
                            if two_sim2>=two_sim1:
                                sim_result=two_sim2
                                left_keywords = st9
                                len_left = len(left_keywords)
                                left_keywords_len.append(len_left)
                                right_keywords = st10
                                len_right = len(right_keywords)
                                right_keywords_len.append(len_right)
                                together_keywords = nr_together_keywords
                                len_together = len(together_keywords)
                                together_keywords_len.append(len_together)
                            else:
                                sim_result = two_sim1
                                left_keywords = st1
                                len_left = len(left_keywords)
                                left_keywords_len.append(len_left)
                                right_keywords = st2
                                len_right = len(right_keywords)
                                right_keywords_len.append(len_right)
                                together_keywords = zy_together_keywords
                                len_together = len(together_keywords)
                                together_keywords_len.append(len_together)

                        except:
                            sim_result = two_sim1
                except:
                    sim_result=0.3
            elif top_zy and not tail_zy:
                try:
                    summary = TextRankSummary()
                    tail_zdzy = ''.join(summary.get_summary_with_textrank(tail))
                    _, tail1 = extract_content()
                    tail1 = tail_zdzy + tail1
                    st3, st4 = Comparison_two(top_zy, tail1)
                    # 转成句子向量
                    vect1 = sent2vec(model, st3)
                    vect2 = sent2vec(model, st4)
                    cos = similarity(vect1, vect2)
                    cos -= 0.1
                    two_sim3 = round(cos, 3)
                    sim_result = two_sim3
                    left_keywords = st3
                    len_left = len(left_keywords)
                    left_keywords_len.append(len_left)
                    right_keywords = st4
                    len_right = len(right_keywords)
                    right_keywords_len.append(len_right)
                    together_keywords = extract_tg_wd(st3 + st4)
                    len_together = len(together_keywords)
                    together_keywords_len.append(len_together)

                except:
                    sim_result = 0.3

            elif not top_zy and tail_zy:
                try:
                    summary = TextRankSummary()
                    top_zdzy = ''.join(summary.get_summary_with_textrank(top))
                    top1, _ = extract_content()
                    top1 = top_zdzy + top1
                    st5, st6 = Comparison_two(top1, tail_zy)
                    # 转成句子向量
                    vect1 = sent2vec(model, st5)
                    vect2 = sent2vec(model, st6)
                    cos = similarity(vect1, vect2)
                    cos -= 0.1
                    two_sim4 = round(cos, 3)
                    sim_result = two_sim4
                    left_keywords = st5
                    len_left = len(left_keywords)
                    left_keywords_len.append(len_left)
                    right_keywords = st6
                    len_right = len(right_keywords)
                    right_keywords_len.append(len_right)
                    together_keywords = extract_tg_wd(st5 + st6)
                    len_together = len(together_keywords)
                    together_keywords_len.append(len_together)

                except:
                    sim_result = 0.3
            else:
                try:
                    summary = TextRankSummary()
                    top_zdzy = ''.join(summary.get_summary_with_textrank(top))
                    tail_zdzy = ''.join(summary.get_summary_with_textrank(tail))
                    top1, tail1 = extract_content()
                    top1 = top_zdzy +top1
                    tail1 = tail_zdzy + tail1
                    st7, st8 = Comparison_two(top1, tail1)

                    # 转成句子向量
                    vect1 = sent2vec(model, st7)
                    vect2 = sent2vec(model, st8)
                    cos = similarity(vect1, vect2)
                    cos -= 0.12
                    two_sim3 = round(cos, 3)
                    sim_result=two_sim3
                    left_keywords = st7
                    len_left = len(left_keywords)
                    left_keywords_len.append(len_left)
                    right_keywords = st8
                    len_right = len(right_keywords)
                    right_keywords_len.append(len_right)
                    together_keywords = extract_tg_wd(st7+st8)
                    len_together = len(together_keywords)
                    together_keywords_len.append(len_together)
                except:
                    sim_result=0.3
            if sim_result<n1:
                continue
            doc_sim={}
            doc_n = 'sim_score'
            doc_sim['Stxt']=[pathes1[i]]
            doc_sim['Ctxt']=[pathes2[i]]
            doc_sim[doc_n] = [str(sim_result)]
            doc_sim['left_keywords_len'] = [left_keywords_len[0]]
            doc_sim['right_keywords_len'] = [right_keywords_len[0]]
            doc_sim['together_keywords_len'] = [together_keywords_len[0]]
            doc_sim['left_keywords'] = [left_keywords]
            doc_sim['right_keywords'] = [right_keywords]
            doc_sim['together_keywords'] = [together_keywords]
            df=pd.DataFrame(doc_sim)
            doc_sim_all=pd.concat([doc_sim_all,df],axis=0)

        # if not doc_sim_all:
        #     return None
        doc_sim_all=doc_sim_all.reset_index(drop=True)

        def drop_dump(data):
            lefts = data['Stxt'].tolist()
            rights = data['Ctxt'].tolist()
            for i,left in enumerate(lefts):
                for j,right in enumerate(rights):
                    if left==right:
                        if lefts[j]==rights[i]:
                            lefts.pop(j)
                            rights.pop(j)
            return lefts,rights

        doc_sim_all1 = doc_sim_all.groupby(['sim_score'])['Stxt', 'Ctxt'].apply(drop_dump).apply(pd.Series).reset_index()
        doc_sim_all1.rename(columns={0: 'Stxt', 1: 'Ctxt'}, inplace=True)
        fenzu_sta1 = doc_sim_all1.set_index(['sim_score'])['Stxt'].apply(
            pd.Series).stack().reset_index()
        fenzu_sta1.drop(['level_1'], axis=1, inplace=True)
        fenzu_sta1.rename(columns={0: 'Stxt'}, inplace=True)

        fenzu_sta2 = doc_sim_all1.set_index(['sim_score'])['Ctxt'].apply(
            pd.Series).stack().reset_index()
        fenzu_sta2.drop(['level_1', 'sim_score'], axis=1, inplace=True)
        fenzu_sta2.rename(columns={0: 'Ctxt'}, inplace=True)
        doc_sim_all1 = pd.concat([fenzu_sta1, fenzu_sta2], axis=1)
        doc_sim_all1 = pd.DataFrame(doc_sim_all1, columns=['Stxt', 'Ctxt', 'sim_score'])
        doc_sim_all1.sort_values('sim_score', ascending=False, inplace=True)
        doc_sim_all=pd.merge(doc_sim_all1,doc_sim_all,on=['Stxt', 'Ctxt', 'sim_score'],how='left')
        doc_sim_all.to_excel('data/相似度.xlsx', index=None, encoding='utf-8')

        path = 'data/相似度.xlsx'
        df = pd.read_excel(path)
        df = df.loc[df['Stxt'] != df['Ctxt']]
        dir = ['D:\czkjlxwj']

        def process(x1):
            try:
                x = x1.strip().split('\\')
                x = x[0] + '\\' + x[1]
                if x not in dir:
                    con = np.nan
                else:
                    con = x1
            except:
                con = np.nan
            return con

        df['Ctxt'] = df.apply(lambda x: process(x['Ctxt']), axis=1)
        df.dropna(subset=['Ctxt'], how='any', axis=0, inplace=True)
        df.to_excel('data/已立项文档相似度比对.xlsx', index=None, encoding='utf-8')

        return doc_sim_all

    #一篇返回十篇
    def doc_sim_(doc):
        result_simlarity =result_sim(doc)
        doc_sim_ = json.dumps(result_simlarity,ensure_ascii=False)  # 字典转json
        return doc_sim_

    #两篇相似度
    def wb_sim_(duibi_doc, doc, n1=0.5):
        wb_sim1 = wb_sim(duibi_doc, doc, n1=0.5)
        wb_sim_ = json.dumps(wb_sim1,ensure_ascii=False)  # 字典转json

        return wb_sim_

    #hash两篇相似度
    def hash_sim_(duibi_doc, doc, n1=0.4):

        hash_sim = wb1_sim(duibi_doc, doc, n1=0.4)
        hash_sim_ = json.dumps(hash_sim,ensure_ascii=False)  # 字典转json

        return hash_sim_

        # 多篇相似度
    def multiple_sim_(path1,path2,n1=0.6):
        path1=eval(path1)
        path2=eval(path2)
        n1=float(n1)
        multi_sim1 = multiple_sim(path1,path2,n1)
        multi_sim_ = json.dumps(multi_sim1, ensure_ascii=False)  # 字典转json

        return multi_sim_

    server.register_function(doc_sim_, 'doc_sim_')
    server.register_function(wb_sim_, 'wb_sim_')
    server.register_function(hash_sim_, 'hash_sim_')
    server.register_function(multiple_sim_, 'multiple_sim_')
    # server.register_instance(vn)
    print("server is start...........")
    # Run the server's main loop
    server.serve_forever()
    print("server is end...........")

