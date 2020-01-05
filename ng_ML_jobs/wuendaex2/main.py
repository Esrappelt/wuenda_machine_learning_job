import tools.cost_grad as cg #自定义模块
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize

def logistic_regression():
    data = loadFile('C:/Users/Administrator/Desktop/ng_ML_jobs/wuendaex2/ex2data2.txt')
    theta,X,y = process_data(data)
    learningRate = 0.01
    res = optimize.fmin_bfgs(cg.costFunction, theta, fprime=cg.gradient, args=(X, y, learningRate))
    p=predict(X,res)      #10.调用预测函数。result参数为我们在上一步中求出的theta最优解
    print('theta的最优解为:',res)
    print('训练的准确度为%f%%'%np.mean(np.float64(p==y)*100))           #p==y实际上是一个bool判断。返回的是一个n行1列的数组，值为False和True,用np.float64转化为0和1数组。
    X = data[:, 0:-1]     #这里需要注意下，重新把X重新定义下，变成只有两个特征的数组。原来的X因为进行了多项式映射，已经有6个了。
    #将结果写入文件，方便下次直接画图
    path = 'C:/Users/Administrator/Desktop/ng_ML_jobs/wuenda2/model/trained_data.txt'
    cg.filein(path,res)
    #画图
    plotBoundry(X,y,res)  #11.画出决策边界 把theta最优解result代入
    return
    
def process_data(data):
    X = np.array(data[...,:-1]) #取每一行，然后取列，一直到最后一列，但不包括最后一列,或者写成X[:,[0,1]],优化数组必须是数组形式
    y = np.array(data[...,-1]) #取最后一列 #优化数组必须是数组形式,这是一维数组,横着的
    X = addBias(X[...,0],X[...,1])
    theta = np.zeros(X.shape[1]) #有5个特征值，加一个偏移项，这是1*6的数组 ,因为优化函数必须是数组形式,横着的
    return [theta,X,y]
    
def addBias(X1,X2):
    X_new = np.ones((X1.shape[0],1)) #这个就是n维向量了,第一列全部为1
    degree = 2
    for i in range(1,degree+1):
        for j in range(i+1):
            x_new = X1 **(i-j) * X2**j
            #hstack就是加一列的意思
            X_new = np.hstack((X_new,x_new.reshape(-1,1))) #reshape(-1,1)的意思是，不知道有多少行，但是最终只转换为一列
    return X_new


def loadFile(path):
    return np.loadtxt(path,delimiter=',',dtype=np.float) #np的loadtxt方法,分隔符位逗号，float型

def predict(X,res):
    m = X.shape[0]
    #X是样本，特征项有6项，x1,x1*x2,x1^2...
    #p就是x与权重theta的乘积之和
    p = cg.sigmoid(np.dot(X,res.T))
    #将p转为0-1数组，大于0.5的就是正类，小于0.5的就是负类
    for i in range(len(p)):
        if(p[i] >= 0.5):
            p[i] = 1
        else:
            p[i] = 0
    return p

def judge(X,y,res):
    p = predict(X,res)
    pre_p = np.mean(np.float64(p==y)*100)
    return pre_p
def plotBoundry(X,y,theta):
    #将0-1类别分开
    y0 = (y==0)
    y1 = (y==1)
    plt.figure(figsize=(6,6))
    plt.plot(X[y0,0],X[y0,1],'b+')
    plt.plot(X[y1,0],X[y1,1],'ro')
    plt.title("decision bondtry")
    u = np.linspace(-1,1.5,50) #从-1到1.5的50的等差数列
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v))) #用来存放每个坐标点上，经过代价函数计算得出的代价值
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = cg.sigmoid(np.dot(addBias(u[i].reshape(1,-1),v[j].reshape(1,-1)),theta))
    z = z.T
    plt.contour(u,v,z,[0,0.5],colors='green')
    plt.show()

logistic_regression()
