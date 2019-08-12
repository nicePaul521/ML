# -*-coding:utf-8-*-
'''
@File   :  regression.py
@Time   :  2019/07/10 21:13:55
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
from numpy import *
import matplotlib.pyplot as plt
from time import sleep
import json
import urllib2
#加载数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1#得到特征数
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat#返回特征集和标签集
#线性回归函数
def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:#判断矩阵是否可逆
        print "This matrix is singular,cann`t do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws
#绘制图形，参数分别为特征集，标签集和回归向量
def plotLine(xArr,yArr,ws):
    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])#绘制数据的散点图
    xCopy = xMat.copy()
    xCopy.sort(0)#表示按列排序
    yHat = xCopy*ws
    ax.plot(xCopy[:,1],yHat)#绘制拟合直线
    plt.show()
#读入数据并返回预测值，参数分别为待预测点，特征集与标签集
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))#创建对角矩阵
    for j in range(m):
        diffMat = testPoint-xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))#得到实例j与待测点之间的权重
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:#判断是否可逆
        print "This matrix is singular,can not do inverse"
        return 
    ws = xTx.I*(xMat.T*(weights*yMat))#得到回归系数向量
    return testPoint*ws#返回待预测点的预测值
#得到待预测集的预测值
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)#得到返回的的预测值
    return yHat#返回预测结果集
#绘制拟合线
def plotLwlr(xArr,yArr):
    xMat = mat(xArr)
    yHat = lwlrTest(xArr,xArr,yArr,0.003)
    srtInd = xMat[:,1].argsort(0)
    print srtInd
    xSort = xMat[srtInd][:,0,:]
    print xSort
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
    plt.show()

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()
#计算回归系数
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular,cannot do inverse"
        return 
    ws = denom.I*(xMat.T*yMat)
    return ws
#用以在一组λ上测试结果
def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)#axis=0，对各列求平均值，返回1*n矩阵
    yMat = yMat - yMean#对yMean广播成yMat大小再执行减运算
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)#求得xMat每个特征的方差，得到1*n矩阵
    xMat = (xMat-xMeans)/xVar#对xMat的每个元素进行标准化
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))#lam以指数级变化
        wMat[i,:] = ws.T
    return wMat#返回不同lam对应的回归系数
#绘制回归系数变化图
def plotRidgeRegres(ridgeWeights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat-yMean
    #xMat = regularize(xMat)
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat-xMeans)/xVar
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    sleep(10)
    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?\key=%s&country=US&q=lego+%d&alt=json'%(myAPIstr,setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['production'] == 'new':
                newFlag=1
            else:newFlag=0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice>origPrc*0.5:
                    print "%d\t%d\t%d\t%f\t%f"%(yr,numPce,newFlag,origPrc,sellingPrice)
                    retX.append([yr,numPce,newFlag,origPrc])
                    retY.append(sellingPrice)
        except:print 'problem with item %d'%i

def setDataController(retX,retY):
    searchForSet(retX,retY,8288,2006,800,49.99)
    searchForSet(retX,retY,10030,2002,3096,269.99)             