# -*-coding:utf-8-*-
'''
@File   :  learningCurve.py
@Time   :  2019/08/07 09:01:08
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
import numpy as np
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt
#here put the import lib
#绘制学习曲线，estimator表示用到的分类器，X是输入的feature，y输入的目标向量，cv作交叉验证集时，数据分成的份数，n_jobs:并行的任务数
def plot_learning_curve(plt,estimator,title,x,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.1,1.0,5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("training examples")
    plt.ylabel("score")
    #依次返回生成learning curve的训练集的样本数，训练集上的分数和测试集上的分数
    train_sizes,train_scores,test_scores = learning_curve(estimator,x,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores,axis=1)#得到训练集和测试集的平均值及标准差
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.grid()#添加网格
    #绘制区域
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o--',color='r',label="training score")#绘制训练数据曲线
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label="Cross-validation score")
    plt.legend(loc="best")
    return plt