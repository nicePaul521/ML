# -*-coding:utf-8-*-
'''
@File   :  sk_svm.py
@Time   :  2019/08/08 15:26:18
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from learningCurve import plot_learning_curve
from sklearn.model_selection import ShuffleSplit
import time

#画出样本点，同时画出分类区间，不同类别的点填充不同的颜色
def plot_hyperplane(clf, X, y, 
                    h=0.02, 
                    draw_sv=True, 
                    title='hyperplan'):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label])
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')

#对线性核函数进行分类
def linear_ker():
    X,Y = make_blobs(n_samples=100,centers=2,random_state=0,cluster_std=0.3)#产生一个具有两个特征，两种类别且大小为100的数据集
    clf = svm.SVC(C=1.0,kernel='linear')#C表示对不符合最大间距的样本点的惩罚力度，此处为线性核
    clf.fit(X,Y)#训练模型
    #绘制分类图
    plt.figure(figsize=(8,3),dpi=144)
    plot_hyperplane(clf,X,Y,h=0.01,title='Maximum Margin Hyperplan')
    plt.show()
#构造4个SVM算法来拟合数据集
def kernel_svm():
    X,Y = make_blobs(n_samples=100,centers=3,random_state=0,cluster_std=0.8)
    clf_linear = svm.SVC(C=1.0,kernel='linear')#定义线性核的分类器
    clf_poly = svm.SVC(C=1.0,kernel='poly',degree=3)#定义度为3的多项式核分类器
    clf_rbf = svm.SVC(C=1.0,kernel='rbf',gamma=0.5)#定义了高斯核分类器
    clf_rbf2 = svm.SVC(C=1.0,kernel='rbf',gamma=0.1)
    plt.figure(figsize=(8,8),dpi=144)
    clfs = [clf_linear,clf_poly,clf_rbf,clf_rbf2]
    titles = ['Linear Kernel','Polynominal Kernel','Gaussian Kernel with $\gamma=0.5$','Gaussian Kernel with $\gamma=0.1$']
    for clf,i in zip(clfs,range(len(clfs))):
        clf.fit(X,Y)
        plt.subplot(2,2,i+1)
        plot_hyperplane(clf,X,Y,title=titles[i])
    plt.show()
#样例：乳腺癌检测
def cancer_detect():
    cancer = load_breast_cancer()
    X = cancer.data
    Y = cancer.target
    print('data shape:{0};no.positive:{1};no.negative:{2}'.format(X.shape,Y[Y==1].shape[0],Y[Y==0].shape[0]))
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    clf = svm.SVC(C=1.0,kernel='rbf',gamma=0.1)
    clf.fit(x_train,y_train)
    train_score = clf.score(x_train,y_train)
    test_score = clf.score(x_test,y_test)
    print('train score:{0};test score:{1}'.format(train_score,test_score))#数据集很小，高斯函数太复杂，容易出现过拟合
    #出现过拟合后，通过选取合适的参数来降低拟合程度
    gammas = np.linspace(0,0.0003,30)
    #构建参数矩阵
    param_grid = {'gamma':gammas}
    clf = GridSearchCV(svm.SVC(),param_grid,cv=5)
    clf.fit(X,Y)
    print('best params:{0};best score:{1}'.format(clf.best_params_,clf.best_score_))
    plt.figure(figsize=(10,4),dpi=144)
    plot_param_curve(plt,gammas,clf.cv_results_,xlabel='$\gamma$')
#绘制参数变化的评分趋势图
def plot_param_curve(plt,params,cv_results,xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_score_mean = cv_results['mean_test_score']
    test_score_std = cv_results['std_test_score']
    #plt.figure(figsize=(6,4),dpi=144)
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(params,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(params,test_score_mean-test_score_std,test_score_mean+test_score_std,alpha=0.1,color='g')
    plt.plot(params,train_scores_mean,'.--',color='r',label="training score")
    plt.plot(params,test_score_mean,'.-',color='g',label="cross valid score")
    plt.legend(loc="best")
    plt.show()
#即使在最优的gammma参数下，平均的分也只有0.936，因此更换为多项式核来优化分类模型
def cancer_detect_opt():
    cancer = load_breast_cancer()
    X = cancer.data
    Y = cancer.target
    #print('data shape:{0};no.positive:{1};no.negative:{2}'.format(X.shape,Y[Y==1].shape[0],Y[Y==0].shape[0]))
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    clf = svm.SVC(C=1.0,kernel='poly',degree=2)
    clf.fit(x_train,y_train)
    train_score = clf.score(x_train,y_train)
    test_score = clf.score(x_test,y_test)
    print('train score:{0};test score:{1}'.format(train_score,test_score))#数据集很小，高斯函数太复杂，容易出现过拟合
    #绘制学习曲线
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    title = 'Learning Curves with degree={0}'
    degrees = [1,2]
    start = time.clock()
    plt.figure(figsize=(10,4),dpi=144)
    for i in range(len(degrees)):
        plt.subplot(1,len(degrees),i+1)
        plot_learning_curve(plt,svm.SVC(C=1.0,kernel='poly',degree=degrees[i]),title.format(degrees[i]),X,Y,ylim=(0.8,1.01),cv=cv,n_jobs=4)
    print('elaspe:{0:.6f}'.format(time.clock()-start))#二阶多项式核函数计算的时间代价很高
    plt.show()