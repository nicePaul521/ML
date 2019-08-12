# -*-coding:utf-8-*-
'''
@File   :  kMeans.py
@Time   :  2019/07/18 22:53:04
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
from numpy import *


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.sprip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centriods = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j])-minJ)
        centriods[:,j] = minJ + rangeJ*random.rand(k,1)
    return centriods