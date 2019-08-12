# -*-coding:utf-8-*-
'''
@File   :  learningCurve.py
@Time   :  2019/08/04 15:57:54
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from matplotlib import  pyplot as plt

n_dots = 200
X = np.linspace(0,1,n_dots)
Y = np.sqrt(X)+0.2*np.random.rand(n_dots)-0.1#添加干扰
X = X.reshape(-1,1)#-1说明行数自动调整，而列数始终为1，调整为样本数*特征数的形式
Y = Y.reshape(-1,1)
#构造一个多项式模型
def polymomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)#得到特征的degree多项式次数
    linear_regression = LinearRegression()
    #这是一个流水线，先增加多项式阶数，在利用线性回归算法拟合数据
    pipeline = Pipeline([("polynomial_features",polynomial_features),("linear_regression",linear_regression)])
    return pipeline#返回一个多项式模型
#绘制学习曲线，estimator表示用到的分类器，X是输入的feature，y输入的目标向量，cv作交叉验证集时，数据分成的份数，n_jobs:并行的任务数
def plot_learning_curve(estimator,title,x,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.1,1.0,5)):
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

#为了让学习曲线更平滑，计算10次交叉验证数据集的分数
def image_show():
    cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
    titles = ['Learning Curves(Under Fitting)','Learning Curves','Learning Curves(Over Fitting)']
    degrees = [1,3,10]
    plt.figure(figsize=(9,4),dpi=200)
    for i in range(len(degrees)):
        plt.subplot(1,3,i+1)
        plot_learning_curve(polymomial_model(degrees[i]),titles[i],X,Y,ylim=(0.75,1.01),cv=cv)
    plt.show()