#!/usr/bin/env python
# -*- coding:utf-8 -*-
import xmlrpc.client
import datetime
import cx_Oracle
import numpy as np
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

def upData(commat):
    connection = cx_Oracle.connect("lbcc", "xdf123", "orcl")
    cursor = connection.cursor()
    cursor.execute(commat)
    connection.commit()
    cursor.close()
    connection.close()
    return

server = xmlrpc.client.ServerProxy("http://127.0.0.1:5008")



user = 'lbcc'
password = 'xdf123'
database = 'orcl'
targetTable1 = 'soure_02'
targetTable2 = 'soure_23'
# commandText1 = '''select t.oraginal_document,t.compare_document,t.ID from SOURE_23 t  where rownum<=50 '''

# train_data = getData(user, password, database, targetTable2, commandText1)
# print(len(train_data))
path='10.24/data4/200tiao.xlsx'
train_data=pd.read_excel(path)
start_time=datetime.datetime.now()
for i in range(len(train_data)):
    oraginal_path = str(train_data['ORAGINAL_DOCUMENT'][i])
    compare_path=str(train_data['COMPARE_DOCUMENT'][i])
    # id=str(train_data['ID'][i])
    commandText2 = '''select t.word from SOURE_02 t where t.file_path='{}' '''.format(oraginal_path)
    commandText3 = '''select t.word from SOURE_02 t  where t.file_path='{}' '''.format(compare_path)
    oraginal_data=getData(user, password, database, targetTable1, commandText2)
    compare_data=getData(user, password, database, targetTable1, commandText3)
    oraginal_content=str(oraginal_data['WORD'][0])
    compare_content=str(compare_data['WORD'][0])
    result = server.hash_sim_(compare_content,oraginal_content)
#     datas=result['data']
#     first_halfs=[]
#     for dict in datas:
#         sim_score=str(dict['sim_score'])
#         Fname=dict['Fname']
#         first_half=path+'\t'+Fname+'\t'+sim_score
#         first_halfs.append(first_half)
#         # writer.write(path+'\t'+json.dumps(result,ensure_ascii=False)+'\n')
    compares=result['compare']
    # second_halfs=[]
    for k,list in enumerate(compares,1):
        try:
            compare_sim=list
            compare_sim=json.dumps(compare_sim,ensure_ascii=False)
            oraginal_id=oraginal_path.split('.')[0].split('\\')[-1]
            compare_id=compare_path.split('.')[0].split('\\')[-1]
            storage_path='D:\\code\\shanghai_text_check\\data5\\{}_{}.json'.format(oraginal_id,compare_id)

            train_data['JSON_PATH'][i]=storage_path
            with open(storage_path,'w',encoding='utf-8') as writer1:
                writer1.write(compare_sim+'\n')
            # commandText4 = '''Merge into {} s  On (s.ID='{}' )
            #                 When matched then update set s.COMPARE_DETAIL = '{}'
            #                '''.format('soure_22', id, storage_path)
            # commandText4 = '''update {} s  set s.COMPARE_DETAIL = '{}' where s.ID='{}'
            #                            '''.format('soure_23', storage_path,id)
            # upData(commandText4)

        except Exception as e:
            print(e)
            compare_sim=''
train_data.to_excel('10.24/data4/top_200.xlsx',index=None,encoding='utf-8')
#     #     second_halfs.append(compare_sim+'\t'+storage_path)
#     # contents=[''.join(tuple)for tuple in zip(first_halfs,second_halfs) if tuple]
#     # writer.write('\n'.join(first_halfs))
#     end_time=datetime.datetime.now()
#     # print(result)
#
#     print('总耗时：%.1f秒' % ((end_time - start_time).seconds))
#     # if j==len(train_data)-1:
#     #     break
#
# df = pd.read_csv('data3/test{}.csv'.format(i),sep='\t')
# # print(df.head())
# df.to_excel('data4/test{}.xlsx'.format(i),index=None,encoding='utf-8')