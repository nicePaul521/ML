# -*-coding:utf-8-*-
'''
@File   :  svmMLiA.py
@Time   :  2019/05/26 14:39:47
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
from numpy import  *
from time import sleep
#打开文件
def loadDataSet(filename):
    dataMat = [];
    labelMat = [];
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return  dataMat,labelMat
#得到一个不等于i的随机索引
def selectJrand(i,m):
    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    return j
#对aj进行剪枝，使其处于[H,L]范围内
def clipAlpha(aj,H,L):
    if aj>H:
        aj = H
    if L>aj:
        aj = L
    return aj
#简化版SMO算法，参数分别为特征矩阵，标签向量，惩罚参数，容忍度，最大迭代次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0;#偏移量
    m,n = shape(dataMatrix)#得到行与列数
    alphas = mat(zeros((m,1)))#初始化alphas为m行1列的0向量
    iter = 0
    while (iter<maxIter):
        alphaPairsChanged = 0#标记alpha对是否已经更新
        for i in range(m):
            #利用分离超平面得到预测值
            fxi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei = fxi - float(labelMat[i])#得到误差值
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                j = selectJrand(i,m)#启发式得到第二个随机变量
                fxj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #固定除i，j之外的其他alpha值，得到最优值alpha的范围
                if (labelMat[i]!=labelMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[i]+alphas[j]-C)
                    H = min(C,alphas[i]+alphas[j])
                if L==H:print("L=H");continue
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print("eta>0")
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta#得到αj沿约束方向更新后的值
                alphas[j] = clipAlpha(alphas[j],H,L)#对αj进行剪枝
                if(abs(alphas[j]-alphaJold)<0.00001):#判断更新程度是否过小
                    print("j not move enough")
                    continue
                alphas[i] += labelMat[i]*labelMat[j]*(alphaJold-alphas[j])#根据更新后的αj求得更新后的αi
                #求得偏移值b
                b1 = b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[j]):
                    b = b1
                elif (alphas[j]>0) and (C>alphas[j]):
                    b = b2
                else:b = (b1+b2)/2.0
                alphaPairsChanged +=1
                print("iter:%d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged))
        if(alphaPairsChanged==0):iter +=1
        else:iter=0
        print("iteration number:%d" % iter)
    return b,alphas
#定义一个保存重要值的结构体类
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))#误差缓存，初始为m行2列的0向量，第一列是ecache是否有效的标志位，第二列是实际的E值
        self.K = mat(zeros((self.m,self.m)))#初始化K位mxm的0矩阵
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)
#计算实例k误差值函数
def calcEk(os,k):
    fXk = float(multiply(os.alphas,os.labelMat).T*os.K[:,k]+os.b)
    Ek = fXk - float(os.labelMat[k])
    return Ek
#内循环中的启发式方法
def selectJ(i,os,Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    #将已经计算好的Ei在缓存中并设置成有效的
    os.eCache[i] = [1,Ei]
    validEcacheList = nonzero(os.eCache[:,0].A)[0]#返回ecache中有效的E的行索引值列表
    if(len(validEcacheList))>1:
        for k in validEcacheList:#遍历索引值列表，找到步长最大的索引值maxK和该索引的误差值Ej
            if k==i:
                continue
            Ek = calcEk(os,k)
            deltaE = abs(Ei-Ek)#得到步长
            if (deltaE>maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:#当第一次循环时缓存中只有i一项是有效的，需随机选择另一个alpha值
        j = selectJrand(i,os.m)
        Ej = calcEk(os,j)
    return j,Ej
#计算误差值并保存到缓存中
def updateEk(os,k):
    Ek = calcEk(os,k)
    os.eCache[k] = [1,Ek]
#内层循环函数
def innerL(i,os):
    Ei = calcEk(os,i)
    if((os.labelMat[i]*Ei<-os.tol) and (os.alphas[i]<os.C)) or ((os.labelMat[i]*Ei>os.tol) and (os.alphas[i]>0)):
        j,Ej = selectJ(i,os,Ei)#找到步长最大的第二个alpha的索引
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        if (os.labelMat[i]!=os.labelMat[j]):
            L = max(0,os.alphas[j]-os.alphas[i])
            H = min(os.C,os.C+os.alphas[j]-os.alphas[i])
        else:
            L = max(0,os.alphas[i]+os.alphas[j]-os.C)
            H = min(os.C,os.alphas[i]+os.alphas[j])
        if L==H:print("L==H");return 0
        eta = 2.0*os.K[i,j]-os.K[i,i]-os.K[j,j]
        if eta>=0:print("eta>0");return 0
        os.alphas[j] -= os.labelMat[j]*(Ei-Ej)/eta
        os.alphas[j] = clipAlpha(os.alphas[j],H,L)
        updateEk(os,j)#更新Ej到缓存并使有效
        if (abs(os.alphas[j]-alphaJold)<0.00001):
            print("j not move enough")
            return 0
        os.alphas[i] +=os.labelMat[i]*os.labelMat[j]*(alphaJold-os.alphas[j])
        updateEk(os,i)#更新Ei到缓存并使有效
        b1 = os.b-Ei-os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,i]-os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[i,j]
        b2 = os.b-Ej-os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,j]-os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[j,j]
        if (0<os.alphas[i]) and (os.C>os.alphas[j]):
            os.b = b1
        elif (os.alphas[j]>0) and (os.C>os.alphas[j]):
            os.b = b2
        else:os.b = (b1+b2)/2.0
        return 1
    else:
        return 0
#完整版Platt SMO算法
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    os = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)#得到结构实例
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):#当大于最大迭代次数或者遍历整个集合都未对alpha对更新则退出循环
        alphaPairsChanged = 0
        if entireSet:#第一个alpha值选自所有数据集
            for i in range(os.m):
                alphaPairsChanged += innerL(i,os)
                print("fullset,iter:%d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged))
                iter +=1
        else:
            nonBoundIs = nonzero((os.alphas.A>0)*(os.alphas.A<C))[0]#两个array相乘是对应位相乘，numpy.A>0返回的是布尔数组
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,os)
                print("non-bound,iter:%d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged)) 
        if entireSet:entireSet = False
        elif (alphaPairsChanged==0):entireSet=True#如果没有alpha对更新，则全局遍历
        print("iteration number:%d"%iter)
    return os.b,os.alphas        
#根据alphas得到w
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
#径向基函数输入
def kernelTrans(X,A,kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin':K = X*A.T
    elif kTup[0]=='rbf':#径向基函数
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))#得到一个k<[0--m-1],A>的数组
    else:
        raise NameError("Houston we have a problem--that Kernel is not recognized")
    return K

def testRbf(kl=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',kl))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors"%shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',kl))
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount +=1
    print("the training error rate is %f"%(float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount=0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',kl))
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount +=1
    print("the test error rate is %f"%(float(errorCount)/m))

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(',')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else:hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s'%(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors"%shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict) != sign(labelArr[i]):
            errorCount +=1
    print("the training error rate is:%f"%(float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount=0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict) != sign(labelArr[i]):
            errorCount +=1
    print("the test error rate is:%f"%(float(errorCount)/m))