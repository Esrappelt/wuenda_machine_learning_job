import numpy as np
from scipy import optimize
import os

def init(theta,X,y):
    #theta是一维向量，X是正常向量，y是n维向量
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
#随机初始,直接利用著名的He初始化方法
def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        x = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['w' + str(l)] = np.insert(x,0, np.zeros(layers_dims[l]),axis=1)
        ### END CODE HERE ###
    print(parameters)
    return parameters

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