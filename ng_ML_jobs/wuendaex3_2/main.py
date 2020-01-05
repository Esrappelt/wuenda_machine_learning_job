import tools.cost_grad as cg
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import classification_report#这个包是评价报告

def loadData():
    path1 = 'C:\\Users\\Administrator\\Desktop\\ng_ML_jobs\\wuenda3_2\ex3data1.mat'
    path2 = 'C:\\Users\\Administrator\\Desktop\\ng_ML_jobs\\wuenda3_2\ex3weights.mat'
    data = loadmat(path1)
    weight = loadmat(path2)
    theta1,theta2 = weight['Theta1'],weight['Theta2']
    X = np.array(data['X'])
    y = np.array(data['y']).flatten()
    X = np.insert(X,0,np.ones(X.shape[0]),axis=1) #axis=1是添加一列
    return [X,y,theta1,theta2]

def bp_net():
    X,y,theta1,theta2 = loadData()
    X,y = np.matrix(X),np.matrix(y).T
    print("X.shape={},y.shape{}".format(X.shape,y.shape))
    print("theta1.shape={},theta2.shape={}".format(theta1.shape,theta2.shape))
    a1 = X
    z2 = a1 * theta1.T
    print("z2.shape={}".format(z2.shape))
    a2 = cg.sigmoid(z2)
    print("a2.shape={}".format(a2.shape))
    a2 = np.insert(a2,0,np.ones(a2.shape[0]),axis=1)
    z3 = a2 * theta2.T
    a3 = cg.sigmoid(z3)
    print(a3)
    print("a3.shape= {}".format(a3.shape))
    ypred = np.argmax(a3,axis=1) + 1 #有10个类，有5000个样本，则求出每一行的最大值，就是哪个类的概率最大，就是哪个类
    print("ypre={}".format(ypred.shape)) #最后是一个5000*1的矩阵，代表着着5000个样本属于哪个类
    print(ypred)
    print(classification_report(y,ypred))
    return
bp_net()