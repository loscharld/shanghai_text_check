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


class Hash_sim:
    def __init__(self):
        pass
    # 将文章每十句合成一个段落
    def func(self,listTemp, n):
        for i in range(0, len(listTemp), n):
            yield listTemp[i:i + n]


    def chaifen_doc(self,doc):
        punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:《》'

        #     with open(doc,encoding='utf-8') as f:
        #         data = f.read()
        list1 = re.split(r"([。!！?？])", doc)
        list1.append("")
        z_list1 = ["".join(i) for i in zip(list1[0::2], list1[1::2])]  # 将文章分成每一句为一个元素存放于列表中

        exam = self.func(z_list1, 10)

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


    def ju_chaifen(self,ju):
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


    def ju_sim(self,duibi_doc, doc):
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

    def wb1_sim(self,duibi_doc,doc,n1=0.4):   #n1:段落相似的设定
    #     duibi_doc = json.loads(duibi_doc)
    #     doc = json.loads(doc)
        db_z_2,db_3,db_4 = self.chaifen_doc(duibi_doc)
        db_z_2 = [i for i in db_z_2 if len(i)>10]
        txt_z_1,txt_1,txt_2 = self.chaifen_doc(doc)
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
        doc_sim = {}
    #     zong_sim_ = {}
        for key,value in doc_index.items():
    #         print(key,value)
            if len(value) >0:
                xuhao = int(key)
                name1 = '第'+str(xuhao+1)+'段'
                for i in value:
                    xuhao_1 = i+1
                    name2 = '与对比文'+str(xuhao_1)+'段'
                    e_1 = self.ju_chaifen(db_3[xuhao])
    #                 print(txt_1[i])
                    e_2 = self.ju_chaifen(txt_1[i])
                    s = self.ju_sim(e_1,e_2)
                    doc_sim[name1+name2+'相似度：'] = str(s)

    #                 print(name1+name2+'相似度：'+str(s))

                    if  len(e_1[-1]) == 0:
                        s_1 =  s*(len(e_1)-1)
                    else:
                        s_1 = s*(len(e_1))
                    sim.append(s_1)

        zong_sim = sum(sim)/len(db_z_2)
        doc_n = '与文章相似度为：'
        doc_sim[doc_n] = str(zong_sim)
        return doc_sim


# 开接口
# class RequestHandler(SimpleXMLRPCRequestHandler):
#     rpc_paths = ('/RPC2123213213',)
#
#
# # Create server
# with SimpleXMLRPCServer(("10.9.1.199", 1112)) as server:
#     def hash_sim_(duibi_doc, doc, n1=0.4):
#         doc_sim = wb_sim(duibi_doc, doc, n1=0.4)
#         doc_sim_ = json.dumps(doc_sim)  # 字典转json
#
#         return doc_sim_
#
#
#     server.register_function(hash_sim_, 'hash_sim_')
#     # server.register_instance(vn)
#     print("server is start...........")
#     # Run the server's main loop
#     server.serve_forever()
#     print("server is end...........")