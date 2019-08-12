# -*-coding:utf-8-*-
'''
@File   :  treeExplore.py
@Time   :  2019/07/16 16:57:03
@Author :  Paul Yu
@Company:  重庆邮电大学
'''
#here put the import lib
from numpy import *
from Tkinter import *
import matplotlib
matplotlib.use('TkAgg')
import regTrees
import sys,os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import  FigureCanvasTkAgg

def reDraw(tolS,tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN<2:tolN=2
        myTree = regTrees.createTree(reDraw.rawDat,regTrees.modelLeaf,regTrees.modelErr,(tolS,tolN))
        yHat = regTrees.createForeCast(myTree,reDraw.testDat,regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawDat,ops = (tolS,tolN))
        yHat = regTrees.createForeCast(myTree,reDraw.testDat)
    print yHat
    reDraw.a.scatter(reDraw.rawDat[:,0].flatten().A[0],reDraw.rawDat[:,1].flatten().A[0],s=5)
    reDraw.a.plot(reDraw.testDat,yHat.flatten().A[0],linewidth=2.0)
    reDraw.canvas.show()

def getInputs():
    try:tolN = int(tolNentry.get())
    except:
        tolN=10
        print('enter integer for tolN')
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    try:tolS = float(tolSentry.get())
    except:
        tolS=1.0
        print('enter float for tolS')
        tolSentry.delete(0,END)
        tolSentry.insert(0,'10')    
    return tolS,tolN

def drawNewTree():
    tolS,tolN = getInputs()
    reDraw(tolS,tolN)

root = Tk()
#reDraw.canvas = Canvas(root,width=500,height=300,bg="white")
#reDraw.canvas.pack()
#reDraw.canvas.grid(row=0,columnspan=3)
reDraw.f = Figure(figsize=(5,4),dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)
Label(root,text="tolN").grid(row=1,column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')
Label(root,text="tolS").grid(row=2,column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'1.0')
Button(root,text="ReDraw",command=drawNewTree).grid(row=1,column=2,rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root,text="Model Tree",variable=chkBtnVar)
chkBtn.grid(row=3,column=0,columnspan=2)
reDraw.rawDat = mat(regTrees.loadDataSet(os.path.join(sys.path[0],'sine.txt')))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
#reDraw(1.0,10)
root.mainloop()