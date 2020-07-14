
import codecs
import re
import sys
import gensim
import sklearn
import numpy as np
import pandas as pd
import jieba
from gensim.models.doc2vec import Doc2Vec,LabeledSentence,TaggedDocument
from jieba import analyse
import jieba.posseg as pesg
import datetime
import os
import time
import json
import cx_Oracle

class Sim_doc2vec:
    def __init__(self):
        # python中对NLS_LANG实现设置
        os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.path_qishu =os.path.join(cur_dir, 'data/train_w2v_10.txt')
        self.path_write =os.path.join(cur_dir, 'similarity_result/doc/' + 'sim_{}.txt'.format(time.time()))
        self.path_dict =os.path.join(cur_dir, 'data/index2id.json')


    def getData(self,user,password,database,targetTable,commandText):
        connection = cx_Oracle.connect(user,password,database)
        cursor = connection.cursor()
        cursor.execute(commandText.format(targetTable))
        x = cursor.description
        columns = [y[0] for y in x]
        cursor01 = cursor.fetchall()
        cursor.close()
        data = pd.DataFrame(cursor01,columns = columns)
        return data

    # def get_dataset(self):
    #     with open(self.path_qishu,'r',encoding='utf-8') as cf:
    #         docs=cf.readlines()
    #     x_train=[]
    #     for i,text in enumerate(docs):
    #         word_list=text.split(' ')
    #         l=len(word_list)
    #         word_list[l-1]=word_list[l-1].strip()
    #         document=TaggedDocument(word_list,tags=[i])
    #         x_train.append(document)
    #     return x_train

    # def train(self,x_train,size=300,epoch_num=1):
    #     model_dm=Doc2Vec(x_train,vector_size=256, window=10, min_count=5,\
    #                                   workers=4, alpha=0.025, min_alpha=0.025, epochs=12)
    #     model_dm.train(x_train,total_examples=model_dm.corpus_count,epochs=12)
    #     model_dm.save('model_doc/model_sim')
    #     return model_dm

    def ceshi(self,text_text):
        model_dm=gensim.models.doc2vec.Doc2Vec.load('model_doc/model_sim')
        inferred_vector_dm=model_dm.infer_vector(text_text.split(' '))
        sims=model_dm.docvecs.most_similar([inferred_vector_dm],topn=10)
        return sims

    def extract(self,line):
        #引用TF_IDF关键词抽取接口
        tfidf=analyse.extract_tags
        str1_fenci=' '.join(jieba.cut(line))
        stop_word=[]
        with open('中文停用词库.txt','r',encoding='utf-8') as fp:
            for line in fp.readlines():
                line=line.strip()
                if line=='':
                    continue
                stop_word.append(line)
        str1_rv_stop_word=''
        str1_rv_stop_word_fenci=''
        for each in str1_fenci.split(' '):
            if each not in stop_word:
                if str1_rv_stop_word=='':
                    str1_rv_stop_word=each
                    str1_rv_stop_word_fenci=each
                else:
                    str1_rv_stop_word=str1_rv_stop_word+each
                    str1_rv_stop_word_fenci=str1_rv_stop_word_fenci+' '+each

        guanjian=tfidf(str1_rv_stop_word_fenci)
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


    def result(self,doc):

        ceshi_list = []
        punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:'
        stoplist = {}.fromkeys([line.rstrip() for line in
                                codecs.open(r"中文停用词库.txt", 'r', 'utf-8')])
        # with open(path, 'r', encoding='utf-8') as f:
        # list1 = f.read()
        string = ''
        X, Y = ['\u4e00', '\u9fa5']
        text1 = re.sub(r'[^\w]+', '', doc)  # 将文本中的特殊字符去掉
        # print(text1)
        s = jieba.cut(text1)
        s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]  # 匹配汉字且不在停用词内的
        string = string.join(s)  # 合并分词
        line1 = re.sub(r"[{}]+".format(punc), "", string)  # 将文本中标点符号去掉
        ceshi_list.append(line1)
        # f1 = open(self.path_write, 'a+', encoding='utf-8')
        for line in ceshi_list:
            # f1.write(line + '\n' + '\n')
            line = self.extract(line)
            sims = self.ceshi(line)
            num=0
            df1 = pd.DataFrame()
            doc2sim = {}
            for count, sim in sims:
                num+=1
                # sentence = x_train[count]
                # words = ''
                # for word in sentence[0]:
                #     words = words + word + ' '
                # print(words, sim, len(sentence[0]))
                with open('data/index2id.json') as f:
                    index2id = json.load(f)
                id=index2id[str(count)]
                user = 'lbcc'
                password = 'xdf123'
                database = 'orcl'
                targetTable = 'soure_01'
                commandText = '''select t.word from SOURE_02 t  where t.id={}
                                                    '''.format(id)
                document=str(self.getData(user,password,database,targetTable,commandText)['WORD'])
                print(document)
                doc2sim[document]=sim

                # df1=pd.concat([df1,document],axis=0)
                # f1.write(str(count) + '\t' + str(sim)+'\t' + document+'\n')
                # print(num)
                # print(sim)
                # print(document)
            # df1.to_csv('data1/result_{}.txt'.format(time.time()),encoding='utf-8',index=None,header=None)
            # f1.write('\n')
        # f1.close()
        return doc2sim

    def Comparison_two(self,fr1,fr2):
        punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:'
        stoplist = {}.fromkeys([line.rstrip() for line in
                                codecs.open(r"中文停用词库.txt", 'r', 'utf-8')])  # 建立字典，词库中的字符为键
        # with open(fr1, 'r', encoding='utf-8') as f:
        #     list1 = f.read()
        string = ''
        X, Y = ['\u4e00', '\u9fa5']
        text1 = re.sub(r'[^\w]+', '', fr1)  # 将文本中的特殊字符去掉
        # print(text1)
        s = jieba.cut(text1)
        s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]  # 匹配汉字且不在停用词内的
        string = string.join(s)  # 合并分词
        line1 = re.sub(r"[{}]+".format(punc), "", string)  # 将文本中标点符号去掉
        word_list1 = [word.word for word in pesg.cut(line1) if word.flag[0] not in ['w', 'x', 'u']]


        # with open(fr2, 'r', encoding='utf-8') as f:
        #     list2 = f.read()
        string = ''
        X, Y = ['\u4e00', '\u9fa5']
        text2 = re.sub(r'[^\w]+', '', fr2)
        # print(text2)
        s = jieba.cut(text2)
        s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]
        string = string.join(s)
        line2 = re.sub(r"[{}]+".format(punc), "", string)
        word_list2 = [word.word for word in pesg.cut(line2) if word.flag[0] not in ['w', 'x', 'u']]
        return word_list1, word_list2

    def sent2vec(self,model, words):
        """文本转换成向量

        Arguments:
            model {[type]} -- Doc2Vec 模型
            words {[type]} -- 分词后的文本

        Returns:
            [type] -- 向量数组
        """

        vect_list = []
        for w in words:
            try:
                vect_list.append(model.wv[w])
            except:
                continue
        vect_list = np.array(vect_list)
        vect = vect_list.sum(axis=0)
        return vect / np.sqrt((vect ** 2).sum())

    def similarity(self,vector1 , vector2):
        """计算两个向量余弦值

        Arguments:
            a_vect {[type]} -- a 向量
            b_vect {[type]} -- b 向量

        Returns:
            [type] -- [description]
        """
        cos1 = np.sum(vector1 * vector2)
        cos21 = np.sqrt(sum(vector1 ** 2))
        cos22 = np.sqrt(sum(vector2 ** 2))
        similarity = cos1 / float(cos21 * cos22)
        return similarity

    def test_model(self,fr1,fr2):
        print("load model")
        model = Doc2Vec.load('model_doc/model_sim')
        st1,st2 = self.Comparison_two(fr1,fr2)
        # 转成句子向量
        vect1 = self.sent2vec(model, st1)
        vect2 = self.sent2vec(model, st2)

        # 查看变量占用空间大小
        # import sys
        # print(sys.getsizeof(vect1))
        # print(sys.getsizeof(vect2))

        cos = self.similarity(vect1, vect2)
        # print("相似度：{:.4f}".format(cos))
        return cos

if __name__=='__main__':
    handle=Sim_doc2vec()
    # x_train =handle.get_dataset()
    with open(handle.path_dict) as f:
        index2id=json.load(f)
    flag = False
    if flag:
        fr1='为突破小样本条件下的快速有效深度学习方法，探索机器学习模型的可解释性机理，我们研究深度神经网络的可解释性表达及可解释性模型的构建。采用图神经网络的方法，实现深度学习的因果推理，实现模型的可解释性。人工智能算法发展的新阶段迫切要求机器能够具有认知智能，实现机器对空间和语义的视觉推理 ，基于常识库实现视觉场景的智能化语义描述，是自动驾驶等领域智能升级的关键技术之一。研究视觉语义理解与视觉推理新算法，视觉场景图构建新算法，基于广义图谱的图卷积网络若干关键技术研究，图注意力网络关键技术研究，图嵌入关键技术研究，图生成网络关键技术， 图空时网络关键技术研究，基于小样本深度模型的四足智能体视觉推理与认知智能关键技术研究。在自动驾驶产业化方面研究的总体目标，是研究自动驾驶中基于可解释深度网络的三维场景感知与理解关键技术，研究基于小样本深度网络的道路特征识别自动驾驶高精度定位技术。'
        fr2='上海市科学技术委员会\n科研计划项目可行性方案\n（ V1.0版）\n指南名称 ？\n上海市2018年度“科技创新行动计划”“一带一路”国际合作项目指\n南\n项目名称 ？ ？tes2t项目国际合作1套\n开始日期 ？2017-07-05\n结束日期 ？2019-10-24\n承担单位 ？ ？网上政务大厅测试机构 （盖章）\n通讯地址 ？ ？通讯地址\n联系电话 ？0210009182 邮政编码200000\n项目责任人 ？王五4\n手机 ？13918811111\n电子邮件 ？ww4@email.com\n2018年07月16日订\n临时\n打印\n 填 写 说 明\n？？？？一、本提纲供编写上海市科学技术委员会科研计划项目可行性方案使用。\n？？？？二、项目承担单位应根据本提纲要求，逐项认真编写，表达要明确严谨，字迹要清楚易辨。外来语同时用\n原文和中文表达。\n？？？？三、申请市科委科研计划项目资助经费在20万元人民币及以下时，毋须填写表2至表 3。\n？？？？所有项目在可行性方案阶段无须填报《表5 实验动物使用情况表》。\n？？？？若项目涉及国际合作事项，则必须填写《表6 国际合作基本信息表》。\n？？？？所有项目必须填写《表7知识产权基本情况表》\n？？？？所有单价≥30万元的购置/试制设备的项目必须填写《表8大型科学仪器设备基本信息表》\n？？？？四、项目申报单位通过上海科技网（http://www.stcsm.gov.cn）上的“科研计划项目可行性方案申报”系统进行相关内容的\n申报，并经申报系统软件打印书面材料（非由申报系统软件打印的书面材料，或书面材料与网上申报材料不一致的项目不予\n受理）。\n？？？？五、报送市科委项目可行性方案书面材料一式一份（特殊情况另定），请使用A4纸双面印刷,请不要采用胶圈、\n文件夹等带有突出棱边的装订方式，请采用普通纸质材料作为封面。\n？？？？六、本提纲制订单位是上海市科学技术委员会。\n？？？？计划类别： 成果转化与应用->战略性新兴产业引导专项->国际科技合作项目->“一带一路”国际合作项目-\n>专题三：“一带一路”技术转移服务领域合作->申报单位为企业\n2\n临时\n打印\n单位（企业）基本情况表\n单位（企业）名称 网上政务大厅测试机构 注册地行政区划 徐汇区\n单位（企业）代码 9 9 9 9 9 9 9 9 - X 电子邮件 yy@email.com\n通 讯 地 址 通讯地址 邮 编 200000\n单位（企业）\n法人代表情况\n姓 名 性 别 身份证号 护照 军官证 最高学历 任现职时间 电话\n姓名 男 310109198011111111 最高学历 任现职时间 0210009181\n联 系 人 0210009181 电 话 0210009182 传 真 0210009183\n科研部门电子邮件 kw@email.com 财务部门电子邮件 cw@email.com\n开户银行 上海市浦东发展银行 开户名 测试账号\n帐 号 123456789098765432\n单位隶属 0 2 01.中央单位 02.地方单位\n注册登记类型\n0 7 01. 国有企业 06. 外商投资企业 11.高等院校\n02. 集体企业 07. 有限责任公司 12.研究院所\n03. 私营企业 08. 股份有限公司 13.社会团体\n04. 联营企业 09. 港、澳、台商投资企业 14.其他\n05. 股份合作企业 10.国家机关\n单位职工总数 10 人 大专以上 2 人 研究开发 2 人\n单位中层以上管理人\n员总数\n3 人 其中大学本科以上人员数 2 人\n企业上年末财务情况，新企业填写申报前一月的财务情况\n企业注册资金 1 万元其中外资（含港澳台）比例 2 ％\n企业注册时间 2014年04月11日\n企业总收入 1 万元 企业净利润 2 万元\n产品销售收入 2 万元 企业创汇总额 2 万美元\n总资产 2 万元 总负债 2 万元\n企业特性\n？ 5 8 ？ (请将下列符合企业情况的代码填入空格内，最多填5项)\n0．国家科技产业化基地内企业 5．科研院所整体转制企业\n？ 1．认定的高新技术企业 6．国家高新区内的企业\n？ 2．高等院校办的企业 7．孵化器内的企业\n3．科研院所办的企业 8．其他\n？ 4．海外归国留学人员办的企业 ？\n？ ？ ？\n单位需要说明的问题：\n？？？？？21\n申报项目 技术领域 ？ 7 9？ (请将下列符合领域情况的代码填入空格内，最多填2项)\n3\n临时\n打印\n1．电子与信息 2.生物、医药 3.先进材料\n4.先进制造 5.资源与环境 6.新能源、高效节能\n7. 绿色农业 8. 其他高新技术 9.软科学研究\n中文关键词 （用分号\n分开，最少3个，最多\n5个）\n？1；1；1；1；1\n英文关键词 （用分号\n分开，最少3个，最多\n5个）\n？1；1；1；1；1\n中文摘要 （限400字） ？？？？？1\n4\n临时\n打印\n项目可行性方案提纲：\n \n？？？？？一、趋势判断和需求分析\n？？？？？国内外现状、水平和发展趋势（含知识产权状况和技术标准状况）；经济建设和社会发展需求；科学技术价值、特色和\n创新点。\n？？？？？2\n？？？？？二、研究内容和技术关键\n？？？？？项目研究的总体目标和创新点，主要内容及所需要解决的技术关键（专利技术二次开发专项申请项目要求着重描述如何\n通过创新形成新的专利技术）。\n？？？？？22\n？？？？？三、执行年限和计划进度\n？？？？？按季度、年度列出计划进度和关键的、必须实现的节点目标。\n？？？？？2\n？？？？？四、工作条件和环境保障\n？？？？？项目申请单位情况；已经具备的实验条件；项目组织机制设计；产学研结合加快工作进展的设想。\n？？？？？2\n？？？？？五、成果形式和考核指标\n？？？？？具体的成果定性、定量考核指标；成果的表达形式，能否申请并获得专利。包括：1.主要技术指标、形成的专利（申请\n不同类别专利数和可望授权专利数）、标准（标准草案和形成的技术标准水平）、新技术、新产品、新装置、论文专著及\n其数量、指标和水平等；2.项目实施中形成的实验室、研发中心、示范基地、中试线、生产线及其规模等；3.经济考核指标\n；4.人才培养情况。\n？？？？？2\n？？？？？六、预期效果和风险分析\n？？？？？项目成果对社会发展所起的作用；经济效益和产业化前景（预计年产值、年利润、年节汇、年创汇、年节能等）；对环\n境影响程度及资源综合利用情况；可能的技术风险；可能的市场风险。\n？？？？？2\n？？？？？七、主要研究人员情况\n？？？？？项目责任人和主要成员简历（学历、工作经历、论著、近三年重要成果及获奖情况等）。\n？？？？？2\n？？？？？八、经费预算\n？？？？？请按《上海市科研计划项目经费预算表》填写。\n？？？？？九、申请者目前承担其他项目资助情况\n？？？？？任务来源、项目编号、项目名称、起止年月、在项目中的责任（项目责任人或参加者）、进展或完成情况。\n5\n临时\n打印\n？？？？？2\n？？？？？十、国内合作形式和合作单位意见\n？？？？？合作单位对合作内容、形式、参加人员数、投入资金数、保证工作条件签署具体意见并盖公章。\n？？？？？2\n？？？？？十一、国际合作内容和合作形式\n？？？？？合作国别，合作内容，合作方式、与国外合作伙伴协议复印件，预期目标。\n？？？？？2\n？？？？？十二、实验动物使用情况（表5）\n？？？？？所有项目在可行性方案阶段无须填报。\n？？？？？2\n？？？？？十三、国际合作基本情况（表6）\n？？？？？确定本项目是否涉及国际合作，若选择是，则必须填报国际合作基本情况表。\n？是否涉及国际合作： 是 否\n？？？？？十四、知识产权情况（表7）\n？？？？？了解项目承担单位知识产权和知识产权管理现有情况，以及项目完成后预计达到的指标情况，申报项目必须填写知识产\n权情况表。\n？？？？？2\n？？？？？十五、附件\n？？？？？包含查新报告、证明文件等。电子文本略附件。\n？？？？？十六、承担单位意见\n？？？？？承担单位法定代表人对项目人员配备及承担单位条件保障、具体自筹经费数的承诺、承诺本项目研究项目内容将不侵犯\n他人知识产权等进行审查并签字，盖单位公章。\n？？？？？2\n6\n临时\n打印\n表1 项目预算表\n金额单位：千元 ？\n序号 科目名称 合计 专项经费 自筹经费 计算依据\n1 一、支出预算 4341.4 1332.4 3009.0 /\n2 （一）直接费用 4336.4 1329.4 3007.0 /\n3 1、设备费 4317.0 1314.0 3003.0 /\n4 （1）购置设备费 4313.0 1312 3001 1\n5 （2）试制设备费 2.0 1 1 1\n6 （3）设备改造与租赁费 2.0 1 1 1\n7 2、材料费 0 0 0 /\n8 3、测试化验加工费 0 0 0 /\n9 4、燃料动力费 0 0 0 /\n10 5、差旅/会议/国际合作与交流费 0 0 0 /\n11\n6、出版/文献/信息传播/知识产权事务\n费\n0 0 0 /\n12 7、劳务费 12.4 12.4 / /\n13 （1）项目责任人 0.4 0.4 / /\n14 （2）项目高级研究人员 0 0 / /\n15 （3）项目参与人员 0 0 / /\n16 （4）引进人才 0 0 / /\n17 （5）临时参与人员 12 12 / /\n18 10、专家咨询费 3.0 1 2 1\n19 11、其他费用 4.0 2 2 1\n20 （二）间接费用 5.0 3 2 1\n21 二、收入预算 4341.4 1332.4 3009.0 /\n22 1、申请从专项经费获得的资助 1332.4 1332.4 / /\n23 2、自筹经费 3009.0 / 3009 上传附件\n预算编制人 （签名） ？ 项目责任人(签名) ？\n财务部门负责人 （签名） ？ 科研管理部门负责人(签名) ？\n注： 1、与本项目有关的前期研究（包括阶段性成果）支出的各项经费不得列入本预算；\n？ 2、申请市科委科研计划项目资助经费在200千元人民币以下时，请在“计算依据”栏中直接作出有关说明；\n7\n临时\n打印\n表4 劳务费预算明细表\n金额单位：千元 ？\n？填表说明： 1.证件类别为：身份证、港澳台通行证及护照；身份证号码为15位、或18位； ？\n？？ 2.性别：若证件类别为“身份证”，则自动获取，其他类别需要填写； ？\n？？ 3.出生日期：年-月-日，例如：1962-01-01；若证件类别为“身份证”，则自动获取，其他类别需要填写出生日期。 ？\n？ 姓名\n证件类\n别\n证件号码 性别 出生日期 现工作单位\n现专业\n技术职务\n目前参\n加其它\n项目\n(课题\n)数/时\n间\n在本项目\n中的责任\n分工\n投入本\n项目的\n计划全\n时工作\n时间\n（人月\n）\n平均资助\n标准（元\n/人月）\n申请专项经费资助额 签章\n？ (1) (2) (3) (4) (5) (6) (7) (8) (9) (10) (11)\n（12）=（10）×（1\n1）/1000\n(13)\n项目责任人 王五4 身份证\n310109198011111\n111\n男 1980-11-11 12 21 1/2 1 2 222 0.4\n临时参与人员 / / / / / / / / / / / 12 /\n累 计 12.4 /\n8\n临时\n打印\n表6 国际合作基本信息表\n（申请市科委科研计划项目无国际合作内容的，毋须填写本表）\n项 目 名 称 tes2t项目国际合作1套\n承 担 单 位 网上政务大厅测试机构\n曾列入何种科技计划（可\n多选）\n国家项目： 重大专项 “863”计划 “973”计划 基础研究计划 其他\n地方项目： ？ ？\n基础研究计划 前沿技术研究发展计划 科技支撑计划\n？ 科技人才培养计划 企业技术创新引导计划\n研发公共服务平台与基地建设\n计划\n？ 科普工程计划 国内合作计划 国际及台港澳合作计划\n？ 软科学研究计划 配套计划 其他计划\n项目研发类型（可多选） 基础研究； 应用研究； 试验发展； 其他\n属何合作协议 （协定）\n政府间科技合作协定，协定名称：\n？ ？ 协定约定期限： 年 月 至 年 月\n其他协议，协定名称：\n？ ？ 协定约定期限： 年 月 至 年 月\n无\n合作方式 （可多选）\n01人员交流；  02信息资料交流；  03技术咨询培训；  04引进技术；  05引进人员\n（来华工作）； 06引进设备；  07分工合作研发；  08国外资源利用； 09其他形式\n项目合作协议\n有，名称：无\n无\n项目合作起止日期 2018 年 07 月 至 2018 年 08 月\n合作国别 合作国别\n省级行政区域\n名称\n省级行政区域名称\n合作\n外方\n机构名称 1\n外方负责人 1 电子邮件 1@email.com\n通讯地址 121\n传 真 02100091 电 话 02100092\n外方\n合作\n投入\n与\n人才\n技术\n引进\n1、经费投入 用于中方： 12 万元（人民币） 用于外方： 2 万元（人民币）\n2、关键技术 引进： 1 项 技术名称 1\n3、关键设备\n投入： 2 项 设备名称 2\n引进： 2 项 设备名称 3\n4、特有资源、信息\n、资料投入\n物种数 样本量 数据量 图纸数 其他（名称：2）\n1 1 1 1 1\n5、人才引进\n计： 11 人 博士后 1 人 博士 1 人 硕士 1 人 技术工程人员 1 人\n计来华工作： 1 人月 其中：正高级职称人员 1 人，来华工作 1 人月\n合作外方主要成员简历\n（学历、工作经历、论著 ？？？？？1\n9\n临时\n打印\n、近三年重要成果及获奖\n情况等） (300字以内)\n合作双方的研究优势互补\n性及合作分工(300字以内)\n？？？？？1\n外方在本合作项目上是否\n获得了所在国项目或经费\n支持 (300字以内)\n？？？？？11\n主要研究目标、内容、合\n作理由(300字以内)\n？？？？？1\n10\n临时\n打印\n 表7 项目承担单位知识产权情况表\n项目计划类别：\n成果转化与应用->战略性新兴产业引导专项->国际科技合作项目->“一带一路”国际合作项目->专题三\n：“一带一路”技术转移服务领域合作->申报单位为企业\n项目名称 tes2t项目国际合作1套\n承担单位 网上政务大厅测试机构\n大类 分类 项目承担前累计数 项目完成后新增数\n专利申请\n外观设计 0 0\n实用新型 0 0\n发明专利 0 0\n专利授权\n外观设计 0 0\n实用新型 0 0\n发明专利 0 0\n软件著作权或集成电路布图\n设计\n版权申请 0 0\n版权登记 0 0\n植物新品种\n申请 0 4\n登记 0 0\n商标\n申请 0 0\n注册 0 0\n上海著名商标 0 0\n中国驰名商标 0 0\n注： 1、所有项目必须填写此表，以便于审核与统计；\n？ 2、项目承担前累计数：指本项目承担前承担单位现有的知识产权情况；\n？\n3、项目完成后新增数：指本项目完成后承担单位预计新增的知识产权情况，不含项目承担前现有的知识产权情况\n；该栏目是本表的主要部分，请认真填写；\n？ 4、知识产权表中以数字填写（无知识产权填零）。\n11\n临时\n打印\n表9 预算说明书\n对各科目支出的主要用途、具体内容及明细支出情况进行详细分析说明，同一支出内容一般不得同时编列不同渠道\n的资金。\n（一）直接费用\n？1、设备费\n？？ （1）购置设备费计算依据\n？？ ？ 1\n？？ （2）试制设备费计算依据\n？？ ？ 1\n？？ （3）设备改造与租赁费计算依据\n？？ ？ 1\n？2、材料费\n？？ /\n？3、测试化验加工费\n？？ /\n？4、燃料动力费\n？？ /\n？5、差旅/会议/国际合作与交流费\n？？ /\n？6、出版/文献/信息传播/知识产权事务费\n？？ /\n？7、劳务费\n？？ /\n？10、专家咨询费\n？？ 1\n？11、其他费用\n？？ 1\n（二）间接费用\n？？ 1\n12\n临时\n打印'
        fr3='txt4/a.txt'
        f3=open(fr3,'r',encoding='utf-8').read()
        two_sim=handle.test_model(fr1,f3)
        print(two_sim)

    else:
        a = datetime.datetime.now()
        # path='txt4'
        # listdir=map(lambda x:os.path.join(path,x),os.listdir(path))
        # for i in listdir:
        i='为突破小样本条件下的快速有效深度学习方法，探索机器学习模型的可解释性机理，我们研究深度神经网络的可解释性表达及可解释性模型的构建。采用图神经网络的方法，实现深度学习的因果推理，实现模型的可解释性。人工智能算法发展的新阶段迫切要求机器能够具有认知智能，实现机器对空间和语义的视觉推理 ，基于常识库实现视觉场景的智能化语义描述，是自动驾驶等领域智能升级的关键技术之一。研究视觉语义理解与视觉推理新算法，视觉场景图构建新算法，基于广义图谱的图卷积网络若干关键技术研究，图注意力网络关键技术研究，图嵌入关键技术研究，图生成网络关键技术， 图空时网络关键技术研究，基于小样本深度模型的四足智能体视觉推理与认知智能关键技术研究。在自动驾驶产业化方面研究的总体目标，是研究自动驾驶中基于可解释深度网络的三维场景感知与理解关键技术，研究基于小样本深度网络的道路特征识别自动驾驶高精度定位技术。'
        handle.result(i)
        b = datetime.datetime.now()
        print("程序运行时间：" + str((b - a).seconds) + "秒")



