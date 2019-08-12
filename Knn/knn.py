# -*-coding:utf-8-*-
'''
@File   :  knn.py
@Time   :  2019/08/04 21:17:16
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.model_selection import ShuffleSplit,learning_curve
from sklearn.feature_selection import SelectKBest
from matplotlib import pyplot as plt
import numpy as np
import sys
import pandas as pd

#生成数据
centers = [[-2,2],[2,2],[0,4]]
x,y = make_blobs(n_samples=60,centers=centers,random_state=0,cluster_std=0.60)

def plot_data():
    plt.figure(figsize=(8,4),dpi=144)
    c = np.array(centers)
    plt.scatter(x[:,0],x[:,1],c=y,s=100,cmap='cool')#画出样本
    plt.scatter(centers[:,0],centers[:,1],s=100,marker='^',c='orange')#画出中心点
    plt.show()

def plot_knn():
    #模型训练
    k = 5
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x,y)#根据特征集和标签集训练得到模型
    #对新数据进行预测
    x_sample = [0,2]
    x_sample = np.array(x_sample).reshape(1,-1)
    print x_sample
    y_sample = clf.predict(x_sample)
    neighbors = clf.kneighbors(x_sample,return_distance=False)#返回了距离预测点最近的k个样本点的索引。
    c = np.array(centers)
    #画出示意图
    plt.figure(figsize=(8,4),dpi=144)
    plt.scatter(x[:,0],x[:,1],c=y,s=100,cmap='cool')#样本
    plt.scatter(c[:,0],c[:,1],s=100,marker='^',c='k')#中心点
    plt.scatter(x_sample[0][0],x_sample[0][1],marker='x',s=100,cmap='cool')#待测点
    for i in neighbors[0]:
        plt.plot([x[i][0],x_sample[0][0]],[x[i][1],x_sample[0][1]],'k--',linewidth=0.6)#预测点与距离最近的5个样本点的连线
    plt.show()    

#使用k近邻算法回归拟合
def plot_knnRegression():
        n_dots = 40
        x = 5*np.random.rand(n_dots,1)#rand得到一个[0,1]上均匀分布的数值，且返回一个n_dots*1的数组
        print np.shape(x)
        y = np.cos(x).ravel()
        #添加噪声
        y += 0.2*np.random.rand(n_dots)-0.1
        k = 5
        knn = KNeighborsRegressor(5)
        knn.fit(x,y)#训练模型，得到拟合曲线
        T = np.linspace(0,5,500)[:,np.newaxis]#对拟合曲线上的点采样，以便绘制拟合曲线
        y_pred = knn.predict(T)
        print knn.score(x,y)#得到拟合曲线对训练样本的拟合准确性
        #画出拟合曲线
        plt.figure(figsize=(8,4),dpi=144)
        plt.scatter(x,y,c='g',label='data',s=100)#绘制样本点
        plt.plot(T,y_pred,c='k',label='prediction',lw=4)#绘制拟合曲线
        plt.axis('tight')#改变坐标轴松弛，确保所有数据点能够恰当显示
        plt.title("KNeighborsRegressions (k=%d)" % k)
        plt.show()

def indians_diabetes_prediction():
        #读取数据
        data = pd.read_csv('F:\machine_learning\Knn\pima-indians-diabetes\diabetes.csv')
        print ('dataset shape {}'.format(data.shape))
        print data.head()
        #将数据切割到特征集和标签集
        X = data.iloc[:,0:8]
        Y = data.iloc[:,8]
        #将样本划分为训练集和测试集，测试集所占样本比例为0.2
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
        #构造三个模型
        models = []
        models.append(("Knn",KNeighborsClassifier(n_neighbors=2)))
        models.append(("KNN with weights",KNeighborsClassifier(n_neighbors=2,weights='distance')))
        models.append(("Radius Neighbors",RadiusNeighborsClassifier(n_neighbors=2,radius=500.0)))
        #分别训练三个模型，并计算评分
        results = []
        for name,model in models:
                kfold = KFold(n_splits=10)#将数据样本分成10组，便于接下来对10组数据的交叉验证
                cv_result = cross_val_score(model,X,Y,cv = kfold)#返回了10次交叉验证的结果，
                results.append((name,cv_result))
        for i in range(len(results)):
                print ("name:{}; cross val score:{}".format(results[i][0],results[i][1].mean()))
        #根据最优模型进行训练
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(x_train,y_train)
        train_score = knn.score(x_train,y_train)
        test_score = knn.score(x_test,y_test)
        print("train score: {}; test score: {}".format(train_score,test_score))
        #绘制该模型的学习曲线
        knn = KNeighborsClassifier(n_neighbors=2)
        cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
        plt.figure(figsize=(10,6),dpi=144)
        plt.title("Learn Curve for KNN Diabetes")
        plt.ylim(0.0,1.1)
        #注意此时learning_curve函数返回的train_score是一个5*10的数组，5表示5次不同比例的训练集，10表示10次交叉验证
        train_sizes,train_scores,test_scores = learning_curve(knn,X,Y,cv=cv,n_jobs=1,train_sizes=np.linspace(.1,1.0,5))
        print np.shape(train_scores)
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
        plt.show()

#特征选择及可视化
def feature_selection():
        #得到一个选择器，选择特征中与输出值相关性最大的k个特征
        selector = SelectKBest(k=2)
        #读取数据
        data = pd.read_csv('F:\machine_learning\Knn\pima-indians-diabetes\diabetes.csv')
        #将数据切割到特征集和标签集
        X = data.iloc[:,0:8]
        Y = data.iloc[:,8]
        X_new = selector.fit_transform(X,Y)
        results = []
        #构造三个模型
        models = []
        models.append(("Knn",KNeighborsClassifier(n_neighbors=2)))
        models.append(("KNN with weights",KNeighborsClassifier(n_neighbors=2,weights='distance')))
        models.append(("Radius Neighbors",RadiusNeighborsClassifier(n_neighbors=2,radius=500.0)))
        #分别训练三个模型，并计算评分
        results = []
        for name,model in models:
                kfold = KFold(n_splits=10)#将数据样本分成10组，便于接下来对10组数据的交叉验证
                cv_result = cross_val_score(model,X_new,Y,cv = kfold)#返回了10次交叉验证的结果，
                results.append((name,cv_result))
        for i in range(len(results)):
                print ("name:{}; cross val score:{}".format(results[i][0],results[i][1].mean()))
        plt.figure(figsize=(8,4),dpi=200)
        plt.ylabel("BMI")
        plt.xlabel("Glucose")
        #画出Y==0的阴性样本，用圆圈表示
        plt.scatter(X_new[Y==0][:,0],X_new[Y==0][:,1],c='r',s=20,marker='o')
        #画出Y==1的阳性样本，用三角表示
        plt.scatter(X_new[Y==1][:,0],X_new[Y==1][:,1],c='g',s=20,marker='^')
        plt.show()