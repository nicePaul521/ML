# -*-coding:utf-8-*-
'''
@File   :  sk_pca.py
@Time   :  2019/08/10 16:50:14
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
import numpy as np
import time
import logging
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

#使用Numpy模拟PCA计算过程
def pca_prcess():
    A = np.array([[3,2000],[2,3000],[4,5000],[5,8000],[1,2000]],dtype='float')
    #数据归一化
    mean = np.mean(A,axis=0)
    norm = A-mean
    #数据缩放
    scope = np.max(norm,axis=0) - np.min(norm,axis=0)
    norm = norm/scope
    #对协方差矩阵奇异值分解
    U,S,V = np.linalg.svd(np.dot(norm.T,norm))#U表示特征矩阵
    #需要把二维数据降成一维，只取特征矩阵的第一列
    U_reduce = U[:,0].reshape(2,1)
    #有了主成分特征矩阵后，就可以对数据进行降维了
    R = np.dot(norm,U_reduce)
    print R#打印降维后的特征
    #按照PCA还原原数据
    Z = np.dot(R,U_reduce.T)
    print np.multiply(Z,scope)+mean
    return A,norm,U,U_reduce,Z
#生成数据处理管道
def std_PCA(**argv):
    scaler = MinMaxScaler()#数据预处理
    pca = PCA(**argv)#PCA降维
    pipeline = Pipeline([('scaler',scaler),('pca',pca)])
    return pipeline
#对数据进行降维和还原，并且画出示意图
def pca_dec_vec(A,norm,U,U_reduce,Z):
    pca = std_PCA(n_components=1)#取了一列特征向量
    R2 = pca.fit_transform(A)#矩阵A经过预处理和PCA降维
    print R2  
    Z2 = pca.inverse_transform(R2) #对降维后的数据进行逆运算，即先进行pca还原，再执行预处理的逆运算
    print Z2
    #绘制降维及恢复示意图
    plt.figure(figsize=(6,6),dpi=144)
    plt.title('physcial meaning of PCA')
    ymin = xmin = -1
    ymax = xmax = 1
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    ax = plt.gca()                                  # gca 代表当前坐标轴，即 'get current axis'
    ax.spines['right'].set_color('none')            # 隐藏坐标轴
    ax.spines['top'].set_color('none')

    plt.scatter(norm[:, 0], norm[:, 1], marker='s', c='b')
    plt.scatter(Z[:, 0], Z[:, 1], marker='o', c='r')
    plt.arrow(0, 0, U[0][0], U[1][0], color='r', linestyle='-')
    plt.arrow(0, 0, U[0][1], U[1][1], color='r', linestyle='-.')
    plt.annotate(r'$U_{reduce} = u^{(1)}$',
                xy=(U[0][0], U[1][0]), xycoords='data',
                xytext=(U_reduce[0][0] + 0.2, U_reduce[1][0] - 0.1), fontsize=10,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate(r'$u^{(2)}$',
                xy=(U[0][1], U[1][1]), xycoords='data',
                xytext=(U[0][1] + 0.2, U[1][1] - 0.1), fontsize=10,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate(r'raw data',
                xy=(norm[0][0], norm[0][1]), xycoords='data',
                xytext=(norm[0][0] + 0.2, norm[0][1] - 0.2), fontsize=10,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate(r'projected data',
                xy=(Z[0][0], Z[0][1]), xycoords='data',
                xytext=(Z[0][0] + 0.2, Z[0][1] - 0.1), fontsize=10,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.show()
    '''
    图中正方形的点是原始数据经过预处理后（归一化，缩放）的数据，圆形的点是从一维恢复到二维后的数据
    '''
#下列是一个人脸识别的例子
def loadDadaset():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    data_home = 'datasets/'
    logging.info('Start to loadset')
    faces = fetch_olivetti_faces(data_home=data_home)#根据pkz文件下载图像数据
    logging.info("Done with load dataset")
    X = faces.data#得到数据集
    Y = faces.target#得到类别目标索引
    targets = np.unique(Y)
    target_names = np.array(["c%d" % t for t in targets])
    n_targets = targets.shape[0]#得到类别数量
    n_samples,h,w = faces.images.shape
    print('sample count:{}\nTarget count:{}'.format(n_samples,n_targets))
    print('Image size:{}x{}\nDataSet shape:{}\n'.format(w,h,X.shape))
#显示照片阵列
def plot_gallery(images,title,h,w,n_row=2)