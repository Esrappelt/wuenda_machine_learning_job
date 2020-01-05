import tools.cost_grad as cg #自定义模块
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.io import loadmat
from sklearn.metrics import classification_report#这个包是评价报告
from sklearn.preprocessing import OneHotEncoder #处理真实值y的
#-------------------------------初始化部分-------------------------------------
#加载数据
def loadData():
    path1 = 'C:\\Users\\Administrator\\Desktop\\ng_ML_jobs\\wuenda4\ex3data1.mat'
    path2 = 'C:\\Users\\Administrator\\Desktop\\ng_ML_jobs\\wuenda4\ex3weights.mat'
    data = loadmat(path1)
    weight = loadmat(path2)
    theta1,theta2 = weight['Theta1'],weight['Theta2']
    X = np.array(data['X'])
    y = np.array(data['y']).flatten()
    X = np.insert(X,0,np.ones(X.shape[0]),axis=1) #axis=1是添加一列
    return X,y,theta1,theta2
#一些初始化，以为高级优化算法的参数做准备
def init(X,y,theta1,theta2):
    #y是n维向量
    X,y,theta1,theta2 = np.matrix(X),np.matrix(y).T,np.matrix(theta1),np.matrix(theta2)
    # print("X.shape={}".format(X.shape))
    # print("y.shape={}".format(y.shape))
    # print("theta1.shape={}".format(theta1.shape))
    # print("theta2.shape={}".format(theta2.shape))
    return X,y,theta1,theta2
#前向传播
def forward_prop(X,theta1,theta2):
    m = X.shape[0]
    a1 = X
    z2 = a1 * theta1.T
    a2 = cg.sigmoid(z2)
    a2 = np.insert(a2,0,np.ones(m),axis=1)
    z3 = a2 * theta2.T
    a3 = cg.sigmoid(z3)
    # print("a1.shape={}".format(a1.shape))
    # print("z2.shape={}".format(z2.shape))
    # print("a2.shape={}".format(a2.shape))
    # print("h.shape={}".format(a3.shape))
    return a1,z2,a2,z3,a3

# 对y标签进行编码 一开始我们得到的y是维的向量，但我们要把他编码成的矩阵。
#  比如说y0 = 2，原始，那么转化后的Y对应行就是[0,1,0...0]，
# 原始转化后的y1 = 0,对应10,对应行就是[0,0...0,1]
# 考虑一下之前的逻辑回归，那个y就只有0，1 不过那是二分类，当多个类时候，就是多个分类器，就要进行转化为2分类
def processY(y):
    encoder = OneHotEncoder(sparse=False)
    newy = encoder.fit_transform(y)
    return newy

#g(z)的导函数
def sigmoid_prime(z):
    #g(z)*(1-g(z))
    return np.multiply(cg.sigmoid(z),1-cg.sigmoid(z))
#手动计算正则化代价函数过程
def computeReg(theta):
    thetaR = theta[...,1:] #除开第一列
    J = 0
    for i in range(thetaR.shape[0]):J += np.dot(thetaR[i],thetaR[i].T)
    return J

#---------------------重要部分-----------------------------------------

#未正则化代价函数
def costFunction(X,y,theta1,theta2):
    a1,z2,a2,z3,h = forward_prop(X,theta1,theta2)
    # 计算代价函数
    J = np.multiply(-y,np.log(h)) - np.multiply(1-y,np.log(1-h))
    #sum的用法，若未告知axis的话，默认是先axis=1相加后，在axis=0相加，就得到和
    #如告知axis则就按axis轴相加
    return np.sum(J)/len(X)

#反向传播
def gradient(X,y,theta1,theta2):
    m = X.shape[0] #样本数量
    a1,z2,a2,z3,h = forward_prop(X,theta1,theta2)
    delta1,delta2 = np.zeros(theta1.shape),np.zeros(theta2.shape) 
    #现在利用初始化theta1和theta2 自动学习最优theta1,theta2
    #开始求解反向传播
    # 输出层的误差
    d3 = h - y
    #隐藏层第一层的误差
    d2 = np.multiply(np.dot(d3,theta2[:,1:]),sigmoid_prime(z2))
    #误差求出来后，开始求解偏导数
    delta1 = np.dot(d2.T,a1)/m
    delta2 = np.dot(d3.T,a2)/m
    return delta1,delta2

#正则化代价函数
def costFunction_Reg(all_theta,X,y,learningRate,input_size,hidden_size,num_labels):
    theta1,theta2 = transform(all_theta,input_size,hidden_size,num_labels)
    #计算正则项，除开第一列
    reg = learningRate/(2 * X.shape[0]) * (np.sum(np.multiply(theta1[...,1:],theta1[...,1:])) + np.sum(np.multiply(theta2[:,1:],theta2[:,1:])))
    # 或者自己手动实现,结果一样的
    # reg = learningRate/(2 * X.shape[0]) * (computeReg(theta1) + computeReg(theta2))
    return reg + costFunction(X,y,theta1,theta2)
#正则化的梯度下降
def gradient_Reg(all_theta,X,y,learningRate,input_size,hidden_size,num_labels):
    m = X.shape[0]
    theta1,theta2 = transform(all_theta,input_size,hidden_size,num_labels)
    delta1,delta2 = gradient(X,y,theta1,theta2)
    delta1[:,1:] += learningRate/m * theta1[:,1:] #对于j>=1的 都要加上正则项,而第0项不加
    delta2[:,1:] += learningRate/m * theta2[:,1:] #对于j>=1的 都要加上正则项,而第0项不加
    grad = np.concatenate((np.ravel(delta1),np.ravel(delta2)),axis=0) #反正就是转化为1维数组,即扁平化
    return grad
#将1维转化为多维,转回去
def transform(all_theta,input_size,hidden_size,num_labels):
    theta1 = np.matrix(np.reshape(all_theta[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(all_theta[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    return theta1,theta2

#----------------------------------主函数部分------------------------------------------------------
def bp_net():
    #获取数据
    X,y,theta1,theta2 = loadData()
    X,y,theta1,theta2 = init(X,y,theta1,theta2)
    y_inital = y
    #这两个theta是文件告诉的theta  本作业是自己初始化theta
    #处理y
    y = processY(y)
    #一些参数
    learningRate = 0.1 #学习率,经测试0.1是最好的
    input_size = 400 #输入层
    hidden_size = 25 #隐藏层1
    num_labels = 10 #输出层
    m = X.shape[0] #样本数量
    #利用著名的he 初始化权重
    params = cg.initialize_parameters_he([input_size,hidden_size,num_labels]) 
    #首先进行theta1和theta2的扁平化，才能作为高级优化函数的参数
    all_theta = np.concatenate((params['w1'].flatten(),params['w2'].flatten()),axis=0) #竖着连接
    #利用高级优化函数求解theta1,theta2
    res = optimize.minimize(fun=costFunction_Reg,x0=all_theta,args=(X,y,learningRate,input_size,hidden_size,num_labels),method='TNC',jac=gradient_Reg,options={'maxiter':800})
    #得出结果
    theta1,theta2 = transform(res.x,input_size,hidden_size,num_labels)
    #进行预测
    _,_,_,_,h = forward_prop(X,theta1,theta2)
    y_pred = np.array(np.argmax(h,axis=1)+1)
    ans = classification_report(y_inital,y_pred)
    return ans
ans = bp_net()
print(ans)

