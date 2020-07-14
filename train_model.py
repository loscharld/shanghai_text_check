import pandas as pd
import re
import codecs
import jieba
import jieba.analyse
import numpy as np
import datetime
import json
import time
# from snownlp import SnowNLP
from gensim import corpora,models,similarities  #用于tf-idf
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models.word2vec import LineSentence, Word2Vec

from collections import defaultdict   #用于创建一个空的字典，在后续统计词频可清理频率少的词语

#连数据库
import cx_Oracle
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
import warnings
import os
from jieba import analyse
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'
# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK'
class Train_model:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.stop_words_path=os.path.join(cur_dir, '中文停用词库.txt')
        self.dict_paths=os.path.join(cur_dir, 'data')
        self.dict_path=os.path.join(self.dict_paths, 'index2id.json')
        self.model_paths = os.path.join(cur_dir, 'model_doc')
        self.model_path=os.path.join(self.model_paths, 'model_sim')

    def getData(self,user, password, database, targetTable, commandText):
        connection = cx_Oracle.connect(user, password, database)
        cursor = connection.cursor()
        cursor.execute(commandText.format(targetTable))
        x = cursor.description
        columns = [y[0] for y in x]
        cursor01 = cursor.fetchall()
        cursor.close()
        data = pd.DataFrame(cursor01, columns=columns)
        return  data

    def extract(self,line):
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
        for each in str1_fenci.split(' '):
            # if each not in stop_word:
            if str1_rv_stop_word=='':
                str1_rv_stop_word=each
                str1_rv_stop_word_fenci=each
            else:
                str1_rv_stop_word=str1_rv_stop_word+each
                str1_rv_stop_word_fenci=str1_rv_stop_word_fenci+' '+each

        guanjian=tfidf(str1_rv_stop_word_fenci,topK=900)
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

    def process_data(self,train_data):
        punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:'
        stoplist = {}.fromkeys([line.rstrip() for line in
                                codecs.open(self.stop_words_path, 'r', encoding='utf-8')])  # 建立字典，词库中的字符为键

        a = datetime.datetime.now()
        count = 0
        sentences = []
        id2index = {}
        for i in range(len(train_data)):
            try:
                list1 = str(train_data['WORD'][i])
                id = str(train_data['ID'][i])
            except Exception as e:
                print(e)
                print('第{}条数据有问题'.format(i))
                continue
            id2index[id] = len(id2index)
            # key_sen = 250
            # t = SnowNLP(list1)
            # t_keysen = t.summary(key_sen)
            # list2=''.join(t_keysen)
            string = ''
            X, Y = ['\u4e00', '\u9fa5']
            text1 = re.sub(r'[^\w]+', '', list1)  # 将文本中的特殊字符去掉
            # print(text1)
            s = jieba.cut(text1)
            s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]  # 匹配汉字且不在停用词内的
            sentence = string.join(s)
            sentence = re.sub(r"[{}]+".format(punc), "", sentence)  # 将文本中标点符号去掉
            sentence=self.extract(sentence)
            sentences.append(sentence)
            count += 1
            # print(count)
        index2id = {v: str(k) for k, v in id2index.items()}
        b = datetime.datetime.now()
        print("数据处理时间：" + str((b - a).seconds) + "秒")
        # with open('data/train_w2v_11.txt', 'w', encoding='utf-8') as writer:
        #     writer.write('\n'.join(sentences))
        return sentences,index2id

    def get_dataset(self,docs):
        x_train=[]
        for i,text in enumerate(docs):
            word_list=text.split(' ')
            l=len(word_list)
            word_list[l-1]=word_list[l-1].strip()
            document=TaggedDocument(word_list,tags=[i])
            x_train.append(document)
        return x_train


    def train(self,x_train,size=300,epoch_num=1):
        model_dm=Doc2Vec(x_train,vector_size=100, window=5, min_count=10,\
                                      workers=4, alpha=0.025, min_alpha=0.025, epochs=12)
        model_dm.train(x_train,total_examples=model_dm.corpus_count,epochs=12)
        model_dm.save(self.model_path)
        return model_dm

    def main(self):
        user = 'lbcc'
        password = 'xdf123'
        database = 'orcl'
        targetTable = 'soure_02'
        commandText = '''select t.id,t.word from SOURE_02 t '''
        if not os.path.exists(handle.model_paths):
            os.mkdir(handle.model_paths)
        if not os.path.exists(handle.dict_paths):
            os.mkdir(handle.dict_paths)
        for i in range(10000):
            start_time = datetime.datetime.now()
            train_data = self.getData(user, password, database, targetTable, commandText)
            print(len(train_data))
            sentences,index2id=self.process_data(train_data)
            x_train=self.get_dataset(sentences)
            print('开始训练模型。。。。。。')
            model_dm = self.train(x_train)
            end_time=datetime.datetime.now()
            print('训练模型总耗时：%.1f秒' % ((end_time - start_time).seconds))
            with open(self.dict_path, 'w') as f:
                json.dump(index2id, f)
            print('Done')
            time.sleep(2592000)

if __name__=='__main__':
    handle = Train_model()
    handle.main()

