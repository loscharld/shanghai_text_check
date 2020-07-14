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

server = xmlrpc.client.ServerProxy("http://127.0.0.1:5002")

# start_time=datetime.datetime.now()
# path1="['D:/code/shanghai_text_check/xinxi3/05302556-X50758-3.txt','D:/code/shanghai_text_check/xinxi6/13212951-958659.txt',\
# 'D:/code/shanghai_text_check/xinxi3/05587768-636932-3.txt','D:/code/shanghai_text_check/xinxi4/00240825-335889-1.txt',\
# 'D:/code/shanghai_text_check/xinxi3/05935817-036139-3.txt','D:/code/shanghai_text_check/xinxi4/05125112-534894-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05125112-556809-1.txt','D:/code/shanghai_text_check/xinxi4/05129771-323836-2.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05295333-X35622-1.txt','D:/code/shanghai_text_check/xinxi4/05297939-636525-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05297939-636525-7.txt']"

# path2="['D:/code/shanghai_text_check/xinxi1/00240825-335889-1.txt',\
# 'D:/code/shanghai_text_check/xinxi1/05121873-955717-1.txt','D:/code/shanghai_text_check/xinxi6/13212951-958659.txt']"

# path3="['D:/code/shanghai_text_check/xinxi1/00240825-335889-1.txt','D:/code/shanghai_text_check/xinxi1/05121873-955717-1.txt',\
# 'D:/code/shanghai_text_check/xinxi1/05125112-534894-1.txt','D:/code/shanghai_text_check/xinxi1/05125112-556809-1.txt',\
# 'D:/code/shanghai_text_check/xinxi1/05295333-X35622-1.txt','D:/code/shanghai_text_check/xinxi1/05297939-636525-1.txt',\
# 'D:/code/shanghai_text_check/xinxi1/05297939-636525-7.txt','D:/code/shanghai_text_check/xinxi1/05304658-X55413-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05295333-X35622-1.txt','D:/code/shanghai_text_check/xinxi4/05297939-636525-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05297939-636525-7.txt']"
#
# path4="['D:/code/shanghai_text_check/xinxi2/00240326-853286-4.txt','D:/code/shanghai_text_check/xinxi2/00240407-636916-4.txt',\
# 'D:/code/shanghai_text_check/xinxi2/00240407-656321-4.txt','D:/code/shanghai_text_check/xinxi2/00240825-332009-4.txt',\
# 'D:/code/shanghai_text_check/xinxi2/00240825-332752-4.txt']"
#
# path5="['D:/code/shanghai_text_check/xinxi3/05302556-X50758-3.txt','D:/code/shanghai_text_check/xinxi3/05587768-636932-3.txt',\
# 'D:/code/shanghai_text_check/xinxi3/05935817-036139-3.txt','D:/code/shanghai_text_check/xinxi3/05935817-050384-3.txt',\
# 'D:/code/shanghai_text_check/xinxi3/06251045-249845-3.txt','D:/code/shanghai_text_check/xinxi2/00240825-332752-4.txt']"
#
# path6="['D:/code/shanghai_text_check/xinxi4/00240326-853286-4.txt','D:/code/shanghai_text_check/xinxi4/00240825-335889-1.txt',\
# 'D:/code/shanghai_text_check/xinxi4/05129771-323836-2.txt','D:/code/shanghai_text_check/xinxi4/05302556-X50758-3.txt',\
# 'D:/code/shanghai_text_check/xinxi4/00240825-332752-4.txt','D:/code/shanghai_text_check/xinxi4/05505885-956017-1.txt']"

path7="['D:/code/shanghai_text_check/xinxi5/05125653-239280-5.txt','D:/code/shanghai_text_check/xinxi5/06939387-739187-5.txt',\
'D:/code/shanghai_text_check/xinxi5/07819977-739338-5.txt','D:/code/shanghai_text_check/xinxi5/09385317-539233-5.txt',\
'D:/code/shanghai_text_check/xinxi5/13220011-939321-5.txt','D:/code/shanghai_text_check/xinxi6/13212951-958659.txt']"

path8="['D:/code/shanghai_text_check/xinxi5/05125653-239280-5.txt','D:/code/shanghai_text_check/xinxi5/06939387-739187-5.txt',\
'D:/code/shanghai_text_check/xinxi5/07819977-739338-5.txt','D:/code/shanghai_text_check/xinxi5/09385317-539233-5.txt',\
'D:/code/shanghai_text_check/xinxi5/13220011-939321-5.txt','D:/code/shanghai_text_check/xinxi6/13212951-958659.txt']"


re=server.multiple_sim_(path8,0.5)
result = json.loads(re)
print(result)