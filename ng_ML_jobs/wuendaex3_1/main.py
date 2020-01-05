import tools.cost_grad as cg
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
import os
from sklearn.metrics import classification_report#这个包是评价报告
#开始写
def logistic_regression():
    #先处理数据
    theta,X,y = process_data(loadData('C:/Users/Administrator/Desktop/ng_ML_jobs/wuenda3_1/ex3data1.mat')) #theta是一维向量，X是正常向量，y是n维向量
    #矩阵的一些参数
    #样本个数
    m = X.shape[0] 
    #theta的维数
    n = X.shape[1]
    #类别个数
    num_labels = 10 
    #分类 ,all_theta,num_label*n的矩阵，每一列代表一个类别
    all_theta = np.zeros((num_labels,n)) 
    #学习率
    alpha = 0.01 
    #求出代价，代价函数
    J = cg.costFunction(theta,X,y,alpha)
    print("J=%f"%J)

    #求解
    for i in range(1,num_labels+1):
        theta = np.zeros(n)
        y_i = np.array([1 if(label == i) else 0 for label in y]) #横着的向量
        grad = cg.gradient(theta,X,y_i,alpha)
        res = optimize.fmin_bfgs(cg.costFunction, theta, fprime=cg.gradient, args=(X, y_i, alpha))
        all_theta[i-1:,...] = res
    path = "C:\\Users\\Administrator\\Desktop\\ng_ML_jobs\\wuenda3\\model\\trained_data.txt"
    #保存文件
    cg.filein(path,all_theta)
    path = "C:\\Users\\Administrator\\Desktop\\ng_ML_jobs\\wuenda3\\model\\trained_data2.txt"
    #进行预测
    y_pred = predict(X,all_theta)
    ans = classification_report(y,y_pred)
    #保存结果
    cg.filein(path,ans)
    print(ans) #打印报告
    return 
def process_data(data):
    #初始时，X正常，y为一维的横着的向量，theta是横着的一维向量
    X = np.array(data['X'])
    y = np.array(data['y']).flatten()
    X = np.insert(X,0,np.ones(X.shape[0]),axis=1) #axis=1是添加一列
    theta = np.zeros(X.shape[1])
    return [theta,X,y]

def loadData(path):
    #打开文件
    data = loadmat(path)
    sample_idx = np.random.choice(np.arange(data['X'].shape[0]),100) #shape[0]就是行的个数,下标从0-100随机选
    sample_images = data['X'][sample_idx,...] #取sample_idx这些行，然后每一列

    #可视化这些图片
    fig,ax_array = plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(12,12))
    for r in range(10):
        for c in range(10):
            ax_array[r,c].matshow(np.array(sample_images[10*r+c].reshape((20,20))).T,cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    # plt.show()
    return data
def predict(X,all_theta):
    num_labels = all_theta.shape[0]
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    #输入一个值，进行权值相乘，经过sigmoid函数进行概率映射
    #这里就取权值最大的那个类，因为，概率越大，就基本为该实例的预测数字标签，概率小就越不是该类
    h = cg.sigmoid(np.dot(X,all_theta.T))
    h_argmax = np.argmax(h,axis=1) #求出最大值,即那个标签
    h_argmax += 1 #标签是加1，因为数字匹配的是0-9
    return h_argmax

logistic_regression()

