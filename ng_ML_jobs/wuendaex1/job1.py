import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#逻辑回归

#数据加载
def dataload():
    data = []
    label = []
    #这里换成自己的文件路径
    with open('C:/Users\Administrator/Desktop/ng_ML_jobs/wuendaex1/ex2data1.txt','r') as f:
        for line in f.readlines():
            tmp = line.strip().split(',')
            data.append([1.0,float(tmp[0]),float(tmp[1])])
            label.append(int(tmp[-1])) #真实值
    return [data,label]

#开始梯度下降
def sigmoid(z):
    return 1/(1+np.exp(-z)) #这里一定要用np.exp,因为会广播

def gradeDescend(data,label):
    data = np.mat(data) #转为矩阵的形式
    label = np.mat(label).T #转为n维向量
    m,n = data.shape #这里是100 * 3的矩阵,即x1,x2,x3 2个特征加一个偏移项
    theta = np.ones((n,1)) #那么theta就是n*1维的向量
    alpha = 0.001 #学习率
    dataT = data.T #计算data的转置
    iterations = 500000 # 500000迭代
    for i in range(iterations):
        predict = sigmoid(data*theta) #预测的值就是数据与权重theta的内积
        loss = predict - label #损失的值就是预测减去真实的值，就是差值，成为误差,注意输出值是label,输入值是data
        theta -= alpha * (data.T * loss) #计算梯度，下降，theta的更新
    return theta
#画图
def logic_plot(theta):
    m = np.shape(data)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(m):
        if(label[i] == 1):
            xcord1.append(data[i][1]) #注意这里是1，不是0，因为第0位是人工加的1
            ycord1.append(data[i][2])
        else:
            xcord2.append(data[i][1])
            ycord2.append(data[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,color='red',label='admit',marker='s',s=30)
    ax.scatter(xcord2,ycord2,color='blue',label='not admit',marker='s',s=30)
    x = np.arange(20,100,2)
    y = (-theta[0]-theta[1] * x ) / theta[2] #画出直线，weights[0]*1.0+weights[1]*x+weights[2]*y=0  
    plt.plot(x,y)
    plt.xlabel('score')
    plt.ylabel('is admitted')
    plt.show()
    return

#评价模型
def hfunc1(theta,x):
    return sigmoid(np.dot(theta.T,x))

[data,label] = dataload()
theta = gradeDescend(data,label)
logic_plot(theta)
p = hfunc1(theta,[1,45,85])
print(p)
