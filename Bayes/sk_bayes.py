# -*-coding:utf-8-*-
'''
@File   :  sk_bayes.py
@Time   :  2019/08/09 10:11:28
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
from time import time
from sklearn.datasets import load_files
import sys,os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

train_path = os.path.join(sys.path[0],'379/train')
test_path = os.path.join(sys.path[0],'379/test')

#将文档转换为向量
def d2v():
    print("loading train datasets...")
    t = time()
    #将词料库加载到内存中
    news_train = load_files(train_path)
    #news_train.data包含了所有文档的文本信息，.target中包含了文档所属的类别，是一个类别索引，.target_names包含了类别名称列表
    print("summary:{0} documents in {1} categories.".format(len(news_train.data),len(news_train.target_names)))
    print("done in {0} seconds".format(time()-t))
    print("vectorizing train datasets...")
    t = time()
    #TfidVectorizer类用来把所有文档转换为矩阵
    vectorizer = TfidfVectorizer(encoding='latin-1')
    #fit_transform()方法将fit与transform()结合，fit会完成词料库分析，提取词典等操作，transform会把每篇文档转换为向量，并构成矩阵
    x_train = vectorizer.fit_transform((d for d in news_train.data))#每一行代表一个文档向量，且该向量含有词料库词数的特征
    print("n_samples:%d,n_features:%d"%x_train.shape)
    #打印第一篇文档非零元素个数，可得这是一个稀疏矩阵
    print("number of non-zero features in sample [{0}]:{1}".format(news_train.filenames[0],x_train[0].getnnz()))
    print("done in {0} seconds".format(time()-t))
    return news_train,x_train,vectorizer
#模型训练
def bayes_train():
    print("train models...")
    t = time()
    News_train,X_train,vc = d2v()
    Y_train = News_train.target
    clf = MultinomialNB(alpha=0.0001)#alpah表示平滑参数，其值越小，越容易造成过拟合，值太大，容易造成欠拟合
    clf.fit(X_train,Y_train)
    train_score = clf.score(X_train,Y_train)
    print("train score:{}".format(train_score))
    print("done in {} seconds".format(time()-t))
    return clf,vc#返回训练模型
#模型测试
def bayes_test(clf,vc):
    print("loading test dataset...")
    t = time()
    #加载测试数据集
    News_test = load_files(test_path)
    print("summary:{0} documents in {1} categories".format(len(News_test.data),len(News_test.target_names)))
    print("done in {0} seconds".format(time()-t))
    print("vectorizing test dataset...")
    t = time()
    #News_train,X_train,vc = d2v()
    x_test = vc.transform((d for d in News_test.data))
    y_test = News_test.target
    print("n_samples:%d,n_features:%d"%x_test.shape)
    print("number of non-zero features in sample [{0}]:{1}".format(News_test.filenames[0],x_test[0].getnnz()))
    print("done in %fs"%(time()-t))
    #对第一篇文章进行验证
    pred = clf.predict(x_test[0])
    print("predict:{0} is in category {1}".format(News_test.filenames[0],News_test.target_names[pred[0]]))
    print("actually:{0} is in category {1}".format(News_test.filenames[0],News_test.target_names[News_test.target[0]]))
    return x_test,y_test,News_test
#模型评估
def model_valid(x_test,y_test,clf,news_test):
    print("predicting test dataset...")
    t0 = time()
    pred = clf.predict(x_test)
    print("done in %fs"%(time()-t0))
    #查看针对每个类别的预测准确性,每种类别统计了查准率，召回率和F1-score
    print("classification report on set for classifier:")
    print(clf)
    print(classification_report(y_test,pred,target_names=news_test.target_names))
    #生成混淆矩阵，观察每种矩阵被错误分类的情况
    cm = confusion_matrix(y_test,pred)
    print("confussion matrix:")
    print(cm)
    #可视化混淆矩阵
    plt.figure(figsize=(8,8),dpi=144)
    plt.title("Confusion matrix of the classifier")
    ax = plt.gca()#获取坐标轴对象
    ax.spines['right'].set_color('none')#设置上下左右坐标轴不可见
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.xaxis.set_ticks_position('none')#设置坐标轴刻度为空
    ax.yaxis.set_ticks_position('none')
    ax.set_xticklabels([])#设置坐标轴标签为空
    ax.set_yticklabels([])
    plt.matshow(cm,fignum=1,cmap='gray')#将矩阵显示，矩阵的值代表颜色强度，此处用gray色系
    plt.colorbar()#显示颜色条
    plt.show()