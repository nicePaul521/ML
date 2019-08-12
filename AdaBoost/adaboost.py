# -*-coding:utf-8-*-
'''
@File   :  adaboost.py
@Time   :  2019/06/12 22:18:56
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
from numpy import *
import matplotlib.pyplot as plt
#加载数据
def loadSimpleData():
    datMat = matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels
#通过阈值比较对数据分类，根据阈值划分为类别+1和-1，并且返回类别向量
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))   
    if threshIneq =='lt':
        retArray[dataMatrix[:,dimen]<=threshVal] = -1.0#某特征值小于阈值的划分为负例
    else:
        retArray[dataMatrix[:,dimen]>threshVal] = -1.0#某特征值大于阈值的划分为负例
    return retArray
#找到数据集上的最佳单层决策树，D表示样本实例的权值向量
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0#步数
    bestStump = {}#决策树属性字典
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):#遍历特征
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps#d得到步长
        for j in range(-1,int(numSteps)+1):#遍历特征值
            for inequal in ['lt','gt']:#遍历不等
                threshVal = (rangeMin+float(j)*stepSize)#得到阈值
                predictVals = stumpClassify(dataMatrix,i,threshVal,inequal)#得到阈值划分的类别向量
                errArr = mat(ones((m,1)))
                errArr[predictVals==labelMat] = 0#将误分类实例置为1
                weightedError = D.T*errArr#得到错误率
                print "split: dim %d, thresh %.2f,thresh ineqal:%s,the weighted error is %.3f"%(i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst#返回最佳决策树的属性信息，最小误差和预测类别向量
#基于单层决策树的Adaboost训练过程，numIt表示迭代次数
def adaBoostTrainDs(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):#会生成numIt个分类器
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#得到分类器的权重alpha
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)#将单层决策树添加到数组中
        print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)#如果正确分类则-α，反之为α
        D = multiply(D,exp(expon))
        D = D/D.sum()#更新样本中实例的权重
        aggClassEst += alpha*classEst#得到多个分类器依权重累加的运算结果
        print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m#统计错误率
        print "total error: ",errorRate,"\n"
        if errorRate == 0:
            break
    return weakClassArr,aggClassEst#返回分类器的数组以及多个分类器组合预测的结果
#adaboost分类函数，参数一表示待分类样例，参数二表示分类器的数组
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):#遍历分类器数组
        #根据分类器中单层决策树的属性对样例进行分类，返回分类结果
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst#计算得分类器累计结果值
        print aggClassEst
    return sign(aggClassEst)#返回预测结果

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
#ROC曲线绘制及AUC计算函数,第一个参数表示分类器的预测强度，第二个参数表示类别向量
def plotROC(predStrengths,classLabels):
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels)==1.0)#得到实际是正例的数目
    yStep = 1/float(max(numPosClas,1e-16))#得到y轴的步长
    xStep = 1/float(max(len(classLabels)-numPosClas,1e-16))#得到x轴步长
    sortedIndicies = predStrengths.argsort()#返回了该列表升序排列的元素的索引
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:#将矩阵转化成列表
        if classLabels[index] == 1.0:#如果是正例则在y轴下降一个步长
            delX = 0
            delY = yStep
        else:#如果是负例则在x轴上倒退一个步长
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve for AdaBoost Horse Colic Detection System")
    ax.axis([0,1,0,1])
    plt.show()
    print "the area under the Curve is :",ySum*xStep