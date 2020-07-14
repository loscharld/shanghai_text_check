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

server = xmlrpc.client.ServerProxy("http://127.0.0.1:5004")

start_time=datetime.datetime.now()
path1='data/所有文档.csv'
df1=pd.read_csv(path1)

# path1='data/200tiao1.xlsx'
# df1=pd.read_excel(path1)
df1=df1.loc[df1['相似度']>=0.6]
df1.drop_duplicates(subset=['原文档','返回文档'],keep='first',inplace=True)
df1=df1.loc[df1['原文档']!=df1['返回文档']]
list1=df1['原文档'].values.tolist()
list2=df1['返回文档'].values.tolist()
# list_all=list(set(list2+list1))
# list_all="{}".format(list_all)
list1="{}".format(list1)
list2="{}".format(list2)
#
# path="['D:/code/shanghai_text_check/xinxi4/00240326-853286-4.txt','D:/code/shanghai_text_check/xinxi4/00240407-636916-4.txt',\
# 'D:/code/shanghai_text_check/xinxi4/00240407-656321-4.txt','D:/code/shanghai_text_check/xinxi4/00240825-332009-4.txt',\
# 'D:/code/shanghai_text_check/xinxi4/00240825-332752-4.txt','D:/code/shanghai_text_check/xinxi4/00240825-335889-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05121873-955717-1.txt','D:/code/shanghai_text_check/xinxi4/05125112-534894-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05125112-556809-1.txt','D:/code/shanghai_text_check/xinxi4/05129771-323836-2.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05295333-X35622-1.txt','D:/code/shanghai_text_check/xinxi4/05297939-636525-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05297939-636525-7.txt']"
re=server.multiple_sim_(list1,list2,0.5)
result = json.loads(re)

print(result)