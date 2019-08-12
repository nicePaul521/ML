# -*-coding:utf-8-*-
'''
@File   :  sk_pca.py
@Time   :  2019/08/10 16:50:14
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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
    Z = np.multiply(Z,scope)+mean
    print Z#打印还原后的特征
    return A
def std_PCA(**argv):
    scaler = MinMaxScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),('pca',pca),('pca',pca)])
    return pipeline

def pca_dec_vec(A):
    pca = std_PCA(n_commponts=1)
    R2 = pca.fit_transform(A)  
    pca.inverse_transform(R2) 