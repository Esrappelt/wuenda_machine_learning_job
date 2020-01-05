import numpy as np
from scipy import optimize
import os

def init(theta,X,y):
    #初始化后theta是一维向量，X是正常向量，y是n维向量
    return [np.matrix(theta),np.matrix(X),np.matrix(y).T]
def sigmoid(z):
    return 1/(1+np.exp(-z))

def gradient(theta, X, y, learningRate):
    theta, X, y = init(theta, X, y)
    #经过初始化后theta是一维向量，X是正常向量，y是n维向量
    loss = sigmoid(np.dot(X,theta.T)) - y    
    grad = np.dot(X.T,loss)/len(X) + (learningRate/len(X)) * theta.T
    grad[0,0] = np.dot(X[...,0].T,loss)/len(X)
    return np.array(grad).ravel()

def costFunction(theta, X, y, learningRate):
    theta, X, y = init(theta, X, y)
    hx = sigmoid(np.dot(X,theta.T))
    J = np.dot(-y.T,np.log(hx))-np.dot((1-y).T,np.log(1-hx)) ##必须是这样，不可以是-np.dot()，这样是错误的
    reg = learningRate / (2 * len(X)) * np.dot(theta,theta.T)
    return reg + J/len(X)
def filein(path,res):
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0], mode=0o777, exist_ok=False)
    try:
        with open(path,'w') as f:
            for key in res:
                f.write(str(key))
                f.write('\n')
        print("成功")
    except Exception as e:
        print('失败',e)
    return 