#!/usr/bin/env python
# -*- coding:utf-8 -*-
import xmlrpc.client
import datetime
import cx_Oracle
import numpy as np
import pandas as pd
import json
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
import os
from jieba import analyse
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'

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

server = xmlrpc.client.ServerProxy("http://127.0.0.1:5003")



user = 'lbcc'
password = 'xdf123'
database = 'orcl'
targetTable = 'soure_01'
commandText = '''select t.file_path,t.word from SOURE_02 t'''
train_data = getData(user, password, database, targetTable, commandText)
print(len(train_data))
start_time=datetime.datetime.now()
for i in range(1,len(train_data)+1):
    with open('data3/test{}.csv'.format(i),'w',encoding='utf-8') as writer:
        writer.write('原文档'+'\t'+'返回文档'+'\t'+'相似度'+'\t'+'比较详情'+'\t'+'路径'+'\n')
        # path_dict={}
        for j in range((i-1)*1,i*1):
            try:
                print(j+1)
                path = str(train_data['FILE_PATH'][j])
                # path_dict['原文档：']=path
                list=str(train_data['WORD'][j])
                result = json.loads(server.doc_sim_(list))
                datas=result['data']
                first_halfs=[]
                for dict in datas:
                    sim_score=str(dict['sim_score'])
                    Fname=dict['Fname']
                    first_half=path+'\t'+Fname+'\t'+sim_score+'\t'
                    first_halfs.append(first_half)
                    # writer.write(path+'\t'+json.dumps(result,ensure_ascii=False)+'\n')
                compares=result['compare']
                second_halfs=[]
                for k,list in enumerate(compares,1):
                    try:
                        compare_sim=list
                        compare_sim=json.dumps(compare_sim,ensure_ascii=False)
                        path_id=path.split('.')[0].split('\\')[-1]
                        storage_path='D:\\code\\shanghai_text_check\\data5\\{}_{}.json'.format(path_id,k)
                        with open(storage_path,'w',encoding='utf-8') as writer1:
                            writer1.write(compare_sim)

                    except Exception as e:
                        print(e)
                        compare_sim=''
                    second_halfs.append(compare_sim+'\t'+storage_path)
                contents=[''.join(tuple)for tuple in zip(first_halfs,second_halfs) if tuple]
                writer.write('\n'.join(contents))
                end_time=datetime.datetime.now()
                # print(result)

                print('总耗时：%.1f秒' % ((end_time - start_time).seconds))
                if j==len(train_data)-1:
                    break
            except Exception as e:
                print(e)
    df = pd.read_csv('data3/test{}.csv'.format(i),sep='\t')
    # print(df.head())
    df.to_excel('data4/test{}.xlsx'.format(i),index=None,encoding='utf-8')