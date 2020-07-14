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

server = xmlrpc.client.ServerProxy("http://127.0.0.1:5009")

# start_time=datetime.datetime.now()
DIR_all='D:\czkjlxwj'
stockpage_dirs=os.listdir(DIR_all)
all_pdfs=[]
for stockpage_dir in  stockpage_dirs:
    stockpage_dir=DIR_all+'\\'+stockpage_dir
    all_pdf=map(lambda _: os.path.join(stockpage_dir, _), os.listdir(stockpage_dir))
    all_pdf = list(filter(lambda _: _.endswith('pdf'), all_pdf))
    all_pdfs.extend(all_pdf)
all_pdfs1="{}".format(list(set(all_pdfs)))
re=server.multiple_sim_(all_pdfs1,0.5)
result = json.loads(re)
print(result)

# '''测试'''
# DIR='D:\czkjlxwj\wjw'
# all_pdf=map(lambda _: os.path.join(DIR, _), os.listdir(DIR))
# all_pdf = list(filter(lambda _: _.endswith('pdf'), all_pdf))
# all_pdfs="{}".format(all_pdf)
# re=server.multiple_sim_(all_pdfs,0.5)
# result = json.loads(re)
# print(result)

# df1=pd.read_excel(path1)
# list1=df1['ORAGINAL_DOCUMENT'].values.tolist()
# list2=df1['COMPARE_DOCUMENT'].values.tolist()
# list_all=list(set(list2+list1))
# list_all="{}".format(list_all)
# list1="{}".format(list1)
# list2="{}".format(list2)
# #
# path="['D:/code/shanghai_text_check/xinxi4/00240326-853286-4.txt','D:/code/shanghai_text_check/xinxi4/00240407-636916-4.txt',\
# 'D:/code/shanghai_text_check/xinxi4/00240407-656321-4.txt','D:/code/shanghai_text_check/xinxi4/00240825-332009-4.txt',\
# 'D:/code/shanghai_text_check/xinxi4/00240825-332752-4.txt','D:/code/shanghai_text_check/xinxi4/00240825-335889-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05121873-955717-1.txt','D:/code/shanghai_text_check/xinxi4/05125112-534894-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05125112-556809-1.txt','D:/code/shanghai_text_check/xinxi4/05129771-323836-2.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05295333-X35622-1.txt','D:/code/shanghai_text_check/xinxi4/05297939-636525-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05297939-636525-7.txt']"
# re=server.multiple_sim_(list1,list2,0.5)
# result = json.loads(re)

# print(result)