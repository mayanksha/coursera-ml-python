import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path

def load_data(filePath):
    p = Path(filePath)
    if(p.is_file()):
        return np.loadtxt(p, delimiter=',')
    else:
        raise ValueError('The file doesnt exists')

def plotter(x_train , y_train):
    y_flat = y_train.flatten()
    plt.scatter(x_train[y_flat==0][:,1], x_train[y_flat==0][:,2], c='red', marker='+')
    plt.scatter(x_train[y_flat==1][:,1], x_train[y_flat==1][:,2], c='green')
    plt.legend()
    plt.xlabel('Exam 1')
    plt.ylabel('Exam 2')

## IN : Takes two arrays as args
def sigmoid(x):
    ans = ((1)/(1+np.exp(-x)))
    # print(ans)
    return ans

## IN : Takes an array as argument (MUST BE NP ARRAY)
def cost_func(theta, x, y):
    m = (x.shape[0])
    h = sigmoid(x.dot(theta))
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    # print(J)
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient_descent(theta, x, y, iterations):
    alpha = 0.002
    m = len(y)
    cost = []
    for i in range(0, iterations):
        cost.append(cost_func(theta, x, y))
        # print('ITERATION NUMBER ====== %d', i)
        theta = theta - (alpha/m)* x.T.dot(sigmoid(np.dot(x,theta)) - y)
        # print((1/m)* x.T.dot(sigmoid(np.dot(x,theta)) - y))
    return (theta, cost)

x_data = (load_data('./ex2data1.txt')[:,0:2])
x_data = np.concatenate((np.ones((100,1)), x_data), axis=1)
y_data = (load_data('./ex2data1.txt')[:,2])
y_data.shape = (100,1)
max_iterations = 400000


theta, cost = gradient_descent(np.zeros((3,1)), x_data, y_data, max_iterations)
print('Working...')
print(theta)

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta)) >= threshold
    return(p.astype('int'))

print(sigmoid(np.array([1, 45, 85]).dot(theta)))
p = predict(theta, x_data) 
plotter(x_data, y_data)

#Choose the minimum and maximum of the Exam1 Data 
x1_min, x1_max = x_data[:,1].min(), x_data[:,1].max(),
#Choose the minimum and maximum of the Exam2 Data 
x2_min, x2_max = x_data[:,2].min(), x_data[:,2].max(),
#Created a meshgrid
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(theta))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');
plt.show()

# print(cost)
