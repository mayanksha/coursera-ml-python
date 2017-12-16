import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

data = np.loadtxt('./ex2data2.txt', delimiter=',')
x = data[:,0:2]
y = data[:,2].reshape((data.shape[0],1))

def plotter(X, Y):
    Y = Y.flatten()
    plt.xlim(-1, 1.5)
    plt.ylim(-1, 1.5)
    plt.scatter(X[Y==0][:,0], X[Y==0][:,1], c='red', marker='+')
    plt.scatter(X[Y==1][:,0], X[Y==1][:,1], c='y' )
    plt.xlabel('test1')
    plt.ylabel('test2')
    plt.show()

def sigmoid(z):
    return (1)/(1 + np.exp(-z))

def costFunction(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta))
    
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1,1)))
      
    grad = (1/m)*XX.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return(grad.flatten())

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    
    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);

poly = PolynomialFeatures(6, include_bias=True)
XX = poly.fit_transform(data[:,0:2])
# initial_theta = np.zeros(XX.shape[1]).reshape((28,1))
# print(costFunction(XX, y, initial_theta, 1))

initial_theta = np.zeros(XX.shape[1])
print(costFunction(initial_theta, 1, XX, y))

#fig is the plt object, axes is an ndarray containing each plot
fig, axes = plt.subplots(1,3, sharey=True, figsize=(17,5))

for i, C in enumerate([0,1,100]):
    res2 = minimize(costFunction, initial_theta, args=(C, XX, y), method=None, jac=gradient, options={'maxiter' : 3000})
    
    accuracy = 100*sum(predict(res2.x, XX) == y.ravel())/y.size    

    # Scatter plot of X,y
    plotData(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
    
    # Plot decisionboundary
    x1_min, x1_max = x[:,0].min(), x[:,0].max(),
    x2_min, x2_max = x[:,1].min(), x[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))

fig.show()
plt.show()
