
# coding: utf-8

from ForCall01 import *
import pandas as pd
import os



os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'



oracle = useOracle("lbcc", "xdf123", "orcl") 



datas=oracle.getData('source_company')


def update_field(comm):
    connection = cx_Oracle.connect("lbcc", "xdf123", "orcl")
    cursor = connection.cursor()
    cursor.execute(comm)
    connection.commit()
    cursor.close()
    connection.close()
    return



import re
for i in range(len(datas)):
    content=str(datas['WORD'][i])
    id=str(datas['ID'][i])
    try:
        company=re.findall('申报单位：(.*?) .*?项目编码',content[:1000].replace('\r\n',' '))[0]
    except:
        pass
    if pd.notnull(company) and len(company)<50:
        com = company
    else:
        try:
            company=re.findall('项目申报单位(.*?)\n',content[0:2000].replace('\r\n',' ').replace('？',''))[0]
        except:
            pass
        if pd.notnull(company) and len(company)<50:
            com=company
        else:
            com=''
    print(com)
    commit='''update source_company t  set t.COMPANY='{0}' where t.ID={1}'''.format(com,id)
    try:
        update_field(commit)
    except:
        continue

