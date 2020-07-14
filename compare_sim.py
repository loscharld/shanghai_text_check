from sim_doc2vec import *


if __name__=='__main__':
    handle = Sim_doc2vec()
    fr1 = 'txt3/5.txt'
    fr2 = 'txt2/24.txt'
    two_sim = handle.test_model(fr1, fr2)
    if float(two_sim) < 0.5:
        print('两篇文章不相似！')
        print('两篇文章相似度为：{}'.format(round(two_sim,3)))
    else:
        print('两篇文章相似！')
        print('两篇文章相似度为：｛｝'.format(round(two_sim, 3)))