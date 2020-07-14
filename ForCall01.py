
# coding: utf-8

# In[7]:


import cx_Oracle
import pandas as pd
from sqlalchemy import create_engine
import time
import numpy as np

class useOracle():
    
    def __init__(self,user,password,database):
        self.user = user
        self.password = password
        self.database = database
        
    def getData(self,targetTable):
        connection = cx_Oracle.connect(self.user, self.password, self.database)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM {}".format(targetTable))
        x = cursor.description
        columns = [y[0] for y in x]
        cursor01 = cursor.fetchall()
        cursor.close()
        data = pd.DataFrame(cursor01,columns = columns)
        return data
    
    def BatchinsertDataToTable(self,data,dataTable):
        connection = cx_Oracle.connect(self.user, self.password, self.database)
        cursor = connection.cursor()
        query = "INSERT INTO "+ dataTable + " VALUES ({})"
        columns = list(data.columns)
        aidx = list(range(1,len(columns)+1))
        aidx = [':'+str(i) for i in aidx]
        aname = ','.join(aidx)
#         print(aname)
        dtHigh = data.shape[0]
        dtWidth = data.shape[1]
        creatVar = locals()
        wholeData = []
        for i in range(dtHigh):
            value_list = []
            for j in range(dtWidth):
                value_list.append("{}".format(str(data.iloc[i,j])))
#             values = ','.join(value_list)
            wholeData.append(value_list)
        cursor.executemany(query.format(aname),wholeData)
        connection.commit()
        cursor.close()
        connection.close()
        return
    
    def getDataThrogPand(self,targetTable):
        engine = create_engine('oracle+cx_oracle://matching:xdf123@LBORA170')
        result = pd.read_sql('select * from {}'.format(targetTable),engine)
#         engine.close()
        return result
    
    def copyTableStructure(self,newTable,targetTable):
        connection = cx_Oracle.connect(self.user, self.password, self.database)
        cursor = connection.cursor()
        cursor.execute("""
                       declare  
                        i integer;  
                        begin  
                        select count(*) into i from user_tables where table_name = '{}';  
                        if i > 0 then  
                        dbms_output.put_line('该表已存在!'); 
                        else  
                        dbms_output.put_line('该表不存在');  
                        execute immediate 'create table {} as select * from {} where 1=2'; 
                        end if;  
                        end;                          
                       """.format(newTable,newTable,targetTable))
#                                 execute immediate 'DROP TABLE {}'; 
        data = self.getData(newTable)
        dHigh = data.shape[0]
        if dHigh == 0:
            print("表格结构已存在，但表格内为空")
            print("表格结构形式为：",data.columns)
        else:
            print("表格已存在，且有数据")
            print("表格大小为：",data.shape)
        print("要删除表格，请执行dropTable命令。")
        cursor.close()
        connection.close()
        return
    
    def insertDataToTable(self,data,dataTable):
        connection = cx_Oracle.connect(self.user, self.password, self.database)
        cursor = connection.cursor()
        query = "INSERT INTO "+ dataTable + " ({}) VALUES ({})"
        columns = list(data.columns)
        aname = ','.join(columns)
        dtHigh = data.shape[0]
        dtWidth = data.shape[1]
        creatVar = locals()
        for i in range(dtHigh):
            value_list = []
            for j in range(dtWidth):
                value_list.append("'{}'".format(str(data.iloc[i,j])))
            values = ','.join(value_list)
            cursor.execute(query.format(aname,values))
        connection.commit()
        cursor.close()
        connection.close()
        return
    
#     def insertDataToTablethrogPand(self,data,tarTable):
#         engine = create_engine('oracle+cx_oracle://{}:{}@{}'.format(self.user, self.password, self.database))
#         data.to_sql('{}'.format(tarTable),con = engine,index=False,if_exists='append')
#         return

    def dropTable(self,dropTable):
        connection = cx_Oracle.connect(self.user, self.password, self.database)
        cursor = connection.cursor()
        cursor.execute("drop table {}".format(dropTable))
        cursor.close()
        connection.close()
        return
    
    def fbPreView(self,table,time):
        connection = cx_Oracle.connect(self.user, self.password, self.database)
        cursor = connection.cursor()
#         print("select * from {} as of timestamp to_timestamp('{}', 'yyyy-mm-dd hh24:mi:ss')".format(table,time))
        cursor.execute("select * from {} as of timestamp to_timestamp('{}', 'yyyy-mm-dd hh24:mi:ss')".format(table,time))
        x = cursor.description
        columns = [y[0] for y in x]
        cursor01 = cursor.fetchall()
        data = pd.DataFrame(cursor01,columns = columns)
        cursor.close()
        connection.close()
        return data
    
    def flashBack(self,table,time):
        connection = cx_Oracle.connect(self.user, self.password, self.database)
        cursor = connection.cursor()
        cursor.execute("alter table {} enable row movement".format(table))
        cursor.execute("flashback table {} to timestamp to_timestamp('{}','yyyy-mm-dd hh24:mi:ss')".format(table,time))
        cursor.close()
        return
    
if __name__ == "__main__":
    oracle = useOracle("repair", "xdf123", "LBORA170") 
    start = time.clock()
    data = oracle.getDataThrogPand("AREA")
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)    
    print(data.shape)  
    oracle.copyTableStructure('BSSV2','BSSV')
    data01 = data.iloc[0:5,:]
    print(data01.shape)
    print(data01.head(2))
    data02 = data01 # .fillna('None')
    aa01 = oracle.insertDataToTable(data02,'BSSV2')
    data03 = oracle.getDataThrogPand('BSSV2')
    print(data03.shape)
    oracle.insertDataToTablethrogPand(data03,'BSSV2')
    data04 = oracle.getDataThrogPand('BSSV2')
    print(data04.shape)
    dropTable = oracle.dropTable('BSSV2')