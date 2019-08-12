# -*-coding:utf-8-*-
'''
@File   :  sk_regression.py
@Time   :  2019/08/06 11:03:49
@Author :  Paul Yu
@Company:  重庆邮电大学
'''

import time
#here put the import lib

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit,learning_curve


def polynormial_model(degree=1):
    polynormial_features = PolynomialFeatures(degree=degree,include_bias=False)#得到多项式的次数，即特征的组合
    linear_regression = LinearRegression(normalize=True)#线性回归分类器
    #得到管道模型，对多项式次数进行拟合的模型，会对输入的数据根据当前模型进行拟合
    pipeline = Pipeline([("polynormial_features",polynormial_features),("linear_regression",linear_regression)])
    return pipeline
#使用线性回归拟合正弦函数
def plot_fit_sin():
    n_dots = 200
    X = np.linspace(-2*np.pi,2*np.pi,n_dots)
    Y = np.sin(X)+0.2*np.random.rand(n_dots)-0.1#添加噪声
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    degrees = [2,3,5,10]#多项式的度列表,此处的度也表示特征多项式的阶数
    results = []
    for d in degrees:
        model = polynormial_model(degree=d)
        model.fit(X,Y)#根据数据训练模型
        train_score = model.score(X,Y)#模型评分
        mse = mean_squared_error(Y,model.predict(X))#得到均方差
        results.append({"model":model,"degree":d,"score":train_score,"mse":mse})
    for r in results:
        print("degree:{}; train score:{} ;mean squard score:{}".format(r["degree"],r["score"],r["mse"]))
    #绘制不同多项式的拟合曲线图
    plt.figure(figsize=(8,4),dpi=200)#subplotpars用于控制下属子图的竖直间距
    plt.subplots_adjust(hspace=0.3)#调整子图竖直间距
    for i,r in enumerate(results):
        fig = plt.subplot(2,2,i+1)
        plt.xlim(-8,8)
        plt.title("LinearRegression degree={}".format(r["degree"]))
        plt.scatter(X,Y,s=5,c='b',alpha=0.5)#数据点散点图
        plt.plot(X,r["model"].predict(X),'r-')#拟合曲线
    plt.show()

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

#测算波士顿房价
def boston_predict():
    boston = load_boston()
    X = boston.data
    Y = boston.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)#将数据集划分为训练集和测试集，测试集占比20%
    model = LinearRegression()
    start = time.clock()#用于计算训练时间
    model.fit(X_train,Y_train)
    train_score = model.score(X_train,Y_train)
    cv_score = model.score(X_test,Y_test)
    print('elapse:{0:.06f}; train score:{1:0.6f}; test score:{2:.6f}'.format(time.clock()-start,train_score,cv_score))
#测算波士顿房价优化
#上个模型欠拟合，因此增加了多项式特征，增加模型复杂度，但是提高了准确率
def boston_opt():
    boston = load_boston()
    X = boston.data
    Y = boston.target
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
    model = polynormial_model(degree=2)
    start = time.clock()
    model.fit(X_train,Y_train)
    train_score = model.score(X_train,Y_train)
    cv_score = model.score(X_test,Y_test)
    print('elapse:{0:.06f}; train score:{1:0.6f}; test score:{2:.6f}'.format(time.clock()-start,train_score,cv_score))
    cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
    title = "Learning Curve(degree={})"
    plt.figure(figsize=(10,4),dpi=200)
    degrees = [1,2,3]
    for i in range(len(degrees)):
        plt.subplot(1,3,i+1)
        plot_learning_curve(plt,polynormial_model(degrees[i]),title.format(degrees[i]),X,Y,ylim=(0.01,1.01),cv=cv)
    plt.show()