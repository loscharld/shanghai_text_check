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


def getData(user, password, database, targetTable, commandText):
    connection = cx_Oracle.connect(user, password, database)
    cursor = connection.cursor()
    cursor.execute(commandText.format(targetTable))
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
    word_list1 = [word.word for word in pesg.cut(line1) if word.flag[0] not in ['w', 'x', 'u']]

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
    word_list2 = [word.word for word in pesg.cut(line2) if word.flag[0] not in ['w', 'x', 'u']]
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


def result_sim(doc):
    doc=json.loads(doc)
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
    doc_sim_all_all=[]
    for key,value in doc_index.items():
#         print(key,value)
        if len(value) >0:
            xuhao = int(key)
            name1 = '原文第'+str(xuhao+1)+'段'

            doc_sim = {}
            for i in value:
                xuhao_1 = i+1
                name2 = '与对比文第'+str(xuhao_1)+'段'
                e1=db_3[xuhao]
                e_1 = ju_chaifen(e1)
#                 print(txt_1[i])
                e2=txt_1[i]
                e_2 = ju_chaifen(e2)
                s = ju_sim(e_1,e_2)
                if float(s) > 0.4:
                    doc_sim[name1] = e1
                    doc_sim[name2] = e2
                    doc_sim['相似度：'] = str(s)
                    doc_sim_all_1.append(doc_sim)
                # doc_sim[name1+name2+'相似度：'] = str(s)
                # doc_sim_all_1.append(doc_sim)

#                 print(name1+name2+'相似度：'+str(s))

                if  len(e_1[-1]) == 0:
                    s_1 =  s*(len(e_1)-1)
                else:
                    s_1 = s*(len(e_1))
                sim.append(s_1)


    if len(doc_sim_all_1) > 5:
        doc_sim_all_1 = doc_sim_all_1[:5]
    zong_sim = sum(sim)/len(db_z_2)
    doc_sim_all_all.append(doc_sim_all_1)
    doc_sim_all['compare'] = doc_sim_all_all

    # doc_sim_all['tscore']=str(zong_sim)
    return doc_sim_all



class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


# Create server
# with SimpleXMLRPCServer(("127.0.0.1", 5002),requestHandler=RequestHandler) as server:
#     server.register_introspection_functions()

# Create server
with ThreadXMLRPCServer(("127.0.0.1", 5008), requestHandler=RequestHandler) as server:
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


    #一篇返回十篇
    def doc_sim_(doc):
        result_simlarity =result_sim(doc)
        doc_sim_ = json.dumps(result_simlarity,ensure_ascii=False)  # 字典转json
        return doc_sim_

    #两篇相似度
    def wb_sim_(duibi_doc, doc, n1=0.5):
        doc_sim = wb_sim(duibi_doc, doc, n1=0.5)
        # doc_sim_ = json.dumps(doc_sim)  # 字典转json

        return doc_sim

    #hash两篇相似度
    def hash_sim_(duibi_doc, doc, n1=0.4):

        doc_sim = wb1_sim(duibi_doc, doc, n1=0.4)
        # doc_sim_ = json.dumps(doc_sim)  # 字典转json

        return doc_sim


    server.register_function(doc_sim_, 'doc_sim_')
    server.register_function(wb_sim_, 'wb_sim_')
    server.register_function(hash_sim_, 'hash_sim_')
    # server.register_instance(vn)
    print("server is start...........")
    # Run the server's main loop
    server.serve_forever()
    print("server is end...........")