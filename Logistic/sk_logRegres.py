# -*-coding:utf-8-*-
'''
@File   :  sk_logRegres.py
@Time   :  2019/08/06 20:27:57
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from learningCurve import plot_learning_curve

cancer = load_breast_cancer()
X = cancer.data
Y = cancer.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

#模型训练
def logregression():
    model = LogisticRegression()#得到逻辑回归模型
    model.fit(X_train,Y_train)#训练模型
    train_score = model.score(X_train,Y_train)
    test_score = model.score(X_test,Y_test)
    print("train score:{:.6f}; test score:{:.6f}.".format(train_score,test_score))
    #样本预测
    y_pred = model.predict(X_test)
    print("matches:{0}/{1}".format(np.equal(y_pred,Y_test).shape[0],Y_test.shape[0]))#预测正确率
#增加多项式预处理
def polynormial_model(degree=1,**kwargs):
    polynormial_features = PolynomialFeatures(degree=degree,include_bias=False)#得到多项式的次数，即特征的组合
    logistic_regression = LogisticRegression(**kwargs)#得到逻辑回归模型
    #得到管道模型，对多项式次数进行拟合的模型，会对输入的数据根据当前模型进行拟合
    pipeline = Pipeline([("polynormial_features",polynormial_features),("logistic_regression",logistic_regression)])
    return pipeline
 #优化后的模型，增加了多项式特征和正则化   
def logregres_opt():
    model = polynormial_model(degree=2,penalty='l1')#将L1范数作为正则项，可实现参数的稀疏化，自动选择出对模型有关联的特征
    start = time.clock()
    model.fit(X_train,Y_train)#训练模型
    train_score = model.score(X_train,Y_train)
    cv_score = model.score(X_test,Y_test)
    print("elapse:{0:.6f}; train score:{1:.6f}; cv_score:{2:.6f}".format(time.clock()-start,train_score,cv_score))
    logistic_regression = model.named_steps['logistic_regression']#得到管道中的逻辑回归模型
    #输出模型的参数，输入特征由30增加到了495个，最终大多数特征被丢弃，只保留了92个有效特征
    print('model parameters shape:{0};count of non-zero element:{1}'.format(logistic_regression.coef_.shape,np.count_nonzero(logistic_regression.coef_)))
#绘制学习曲线
def plot_logregres():
    cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
    title = "Learning Curve(degree={0},penalty={1})"
    degrees = [1,2]
    penalty = 'l1'
    plt.figure(figsize=(8,4),dpi=144)
    for i in range(len(degrees)):
        plt.subplot(1,len(degrees),i+1)
        plot_learning_curve(plt,polynormial_model(degree=degrees[i],penalty=penalty),title.format(degrees[i],penalty),X,Y,ylim=(0.8,1.01),cv=cv)
    plt.show()
