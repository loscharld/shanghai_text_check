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

content='''织物外观平整度指标反映了服装洗后的面料形状保持性，广泛用于纺织服装的材料性能、织物整
理效果、洗涤剂及洗涤设备性能等的评估。目前国内外对织物外观平整度的评定多依赖人工视觉主观
检测，导致结果精度低、重复性差、偏差较大、检测成本高、效率低等问题。本项目拟基于机器视觉
技术的客观评价技术，研制图像采集装置，建立织物外观三维重建的有效算法，实现织物立体视觉重
建，构建出织物外观平整度视觉字典，实现机器的智能评价；测试代表性织物洗涤、烘干平整度变化
规律，筛选符合敏感性、稳定性、时效性要求的标准织物；研究高度主客观一致性的织物外观平整度
评价算法，实现大样本织物平整度高精度智能评级系统；建立并发布2项技术标准。本项目成果将用
于家用洗涤、烘干设备平整度性能客观评价，解决主观评价的上述痛点，提升我国家电行业与纺织行
业相关产品的质量水平。
'''

re=server.doc_sim_(content)
result = json.loads(re)
print(result)