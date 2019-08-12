# -*-coding:utf-8-*-
'''
@File   :  sk_dt.py
@Time   :  2019/08/07 10:45:25
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
import sys,os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import GridSearchCV

file_path = os.path.join(sys.path[0],r"titanic\train.csv")
#数据清洗
def read_dataset(fname):
    #指定第一列作为行索引
    data = pd.read_csv(fname,index_col=0)
    #丢弃无用的数据
    data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)#删除包含缺失值的行，inplace表示是否在原数据上操作
    #处理性别数据
    data['Sex'] = (data['Sex']=='male').astype('int')#将布尔值转换成int类型，此处男性为1，女性为0
    #处理登船港口的数据
    labels = data['Embarked'].unique().tolist()#去重显示该列的数值，并转化为列表类型
    data['Embarked'] = data['Embarked'].apply(lambda n:labels.index(n))#得到数值n在labels中对应的索引值
    #处理缺失数据
    data = data.fillna(0)#将缺失值填充为0
    return data
#读取处理后的训练数据
train = read_dataset(file_path)
Y = train['Survived'].values
X = train.drop(['Survived'],axis=1).values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
#print('train shape:{}; test shape:{}'.format(X_train.shape,X_test.shape))#打印训练与测试数据集的规模
#训练决策树
def dt_train():
    clf = DecisionTreeClassifier()#决策分类树模型
    clf.fit(X_train,Y_train)#训练模型
    train_score = clf.score(X_train,Y_train)
    test_score = clf.score(X_test,Y_test)
    print("train score:{}; test score:{}".format(train_score,test_score))#根据结果判断出过拟合的特征
#优化模型参数max_depth,即指定了决策树的最大深度
def cv_score1(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train,Y_train)
    tr_score = clf.score(X_train,Y_train)
    cv_score = clf.score(X_test,Y_test)
    return (tr_score,cv_score)
#测试模型参数找到评分最高模型对应的参数
def max_depth_test():
    depths = range(2,15)
    scores = [cv_score1(d) for d in depths]#得到评分结果的元组列表
    tr_scores = [s[0] for s in scores]#得到训练评分集
    cv_scores = [s[1] for s in scores]#得到测试评分集
    #找出交叉验证数据集评分最高的索引
    best_score_index = np.argmax(cv_scores)#返回测试评分最高的索引
    best_score = cv_scores[best_score_index]
    best_param = depths[best_score_index]
    print('best param:{}; best score:{};'.format(best_param,best_score))
    plt.figure(figsize=(6,4),dpi=144)
    plt.grid()
    plt.xlabel('max depth of decision tree')
    plt.ylabel('score')
    plt.plot(depths,cv_scores,'.g--',label='cross-validation score')
    plt.plot(depths,tr_scores,'.r--',label='training score')
    plt.legend()
    plt.show()
#优化模型参数min_impurity_split，用于指定信息熵或基尼不纯度的阈值
#训练模型并评分
def cv_score2(val):
    clf = DecisionTreeClassifier(criterion='gini',min_impurity_decrease=val)
    clf.fit(X_train,Y_train)
    tr_score = clf.score(X_train,Y_train)
    cv_score = clf.score(X_test,Y_test)
    return tr_score,cv_score
#指定参数范围，分别训练模型并计算评分
def min_impurity_test():
    values = np.linspace(0,0.5,50)
    scores = [cv_score2(d) for d in values]#得到评分结果的元组列表
    tr_scores = [s[0] for s in scores]#得到训练评分集
    cv_scores = [s[1] for s in scores]#得到测试评分集
    #找出交叉验证数据集评分最高的索引
    best_score_index = np.argmax(cv_scores)#返回测试评分最高的索引
    best_score = cv_scores[best_score_index]
    best_param = values[best_score_index]
    print('best param:{}; best score:{};'.format(best_param,best_score))
    #画出模型参数与模型评分的关系
    plt.figure(figsize=(6,4),dpi=144)
    plt.grid()
    plt.xlabel('max depth of decision tree')
    plt.ylabel('score')
    plt.plot(values,cv_scores,'.g--',label='cross-validation score')
    plt.plot(values,tr_scores,'.r--',label='training score')
    plt.legend()
    plt.show()
#对于以上参数选择的不稳定和每次只能选择一个参数的情况，进行优化
def param_select_opt():
    thresholds = np.linspace(0,0.5,50)
    #设置参数矩阵
    param_grid = {'min_impurity_split':thresholds}
    try:
    #枚举列表中的所有值构建模型，多次计算训练模型，参数用于交叉验证数据集的生成规则
        clf = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5,return_train_score=True)
        clf.fit(X,Y)
    except Exception as e:
        pass
    #best_params_中保存最优参数，最优评分保存在best_score_,cv_results_保存了计算过程中的所有中间结果
    print('best param:{}\nbest score:{}'.format(clf.best_params_,clf.best_score_))
    plot_curve(thresholds,clf.cv_results_,xlabel='gini thresholds')
#绘制模型参数与评分的关系
def plot_curve(train_sizes,cv_results,xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_score_mean = cv_results['mean_test_score']
    test_score_std = cv_results['std_test_score']
    plt.figure(figsize=(6,4),dpi=144)
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_score_mean-test_score_std,test_score_mean+test_score_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'.--',color='r',label="training score")
    plt.plot(train_sizes,test_score_mean,'.-',color='g',label="cross valid score")
    plt.legend(loc="best")
    plt.show()
#在多组参数之间选择最优的参数
def multiply_param_select():
    entropy_thresholds = np.linspace(0,1,50)
    gini_threshold = np.linspace(0,0.5,50)
    #设置参数矩阵
    param_grid = [{'criterion':['entropy'],'min_impurity_split':entropy_thresholds},
                    {'criterion':['gini'],'min_impurity_split':gini_threshold},
                    {'max_depth':range(2,20)},
                    {'min_samples_split':range(2,30,2)}]
    clf = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
    clf.fit(X,Y)
    print("best param:{};\nbest score:{};".format(clf.best_params_,clf.best_score_))
#生成决策树图形
def dt_generate():
    clf = DecisionTreeClassifier(criterion='entropy',min_impurity_split=0.53061)
    clf.fit(X_train,Y_train)
    train_score = clf.score(X_train,Y_train)
    test_score = clf.score(X_test,Y_test)
    print('train score:{0};test score{1}'.format(train_score,test_score))
    #导出titanic.dot文件
    with open("titanic.dot", 'w') as f:
        f = export_graphviz(clf, out_file=f)
    #使用graphviz工具包将.dot文件转换为png图片