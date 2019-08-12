# -*-coding:utf-8-*-
'''
@File   :  utils.py
@Time   :  2019/08/08 18:02:54
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
#此模块包含了笔者自定义的诸多常用方法，供其他模块调用
import numpy as np
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt

#绘制学习曲线，estimator表示用到的分类器，X是输入的feature，y输入的目标向量，cv作交叉验证集时，数据分成的份数，n_jobs:并行的任务数
'''
plt:画板
estimator:分类器模型
title：绘制图形标题
x,y:特征集与标签集
ylim：纵坐标范围
cv：交叉验证参数
n_jobs:并行的任务数
train_sizes:横坐标的坐标点，表示样本中训练数据集的百分比，此处有五个标记点
'''
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

#画出样本点，同时画出分类区间，不同类别的点填充不同的颜色，用于支持向量机的样本分割结果的绘制
'''
clf:分类器模型
X,y:特征集与标签集
h:
draw_sc:是否显示支持向量
title：图像标题
'''
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
#绘制参数变化的评分趋势图
'''
plt:画板
params：参数值列表
cv_results:由GridSearchCV寻找最优参数过程产生的结果集
xlabel：x轴标签
'''
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