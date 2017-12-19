import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression as LR

import seaborn as sns

def sigmoid(z):
    return (1)/(1 + np.exp(-z))


## theta.shape = (401,1)
def costFunc(theta, reg, X, y):
    theta = theta.flatten().reshape(-1,1)
    m = X.shape[0]
    h = sigmoid(X.dot(theta)).reshape(-1,1)
    J = (-1/m)*((np.log(h)).T.dot(y) + np.log(1-h).T.dot(1-y)) + (reg/(2*m)) * theta.T.dot(theta)
    # print("INSIDE COST FUNCTION")
    # print('h shape = ', h.shape)
    # print('theta shape = ', theta.shape)
    # print('y shape = ', y.shape)
    # print('J shape = ', J.shape)
    # print('(h-y) shape = ', (h-y).shape)
    return J

def gradient(theta, reg, X, y):
    theta = theta.flatten().reshape(-1,1)
    m = X.shape[0]
    h = sigmoid(X.dot(theta)).reshape(-1,1)
    grad = (1/m)*(X.T.dot(h-y)) + (reg/m)*(np.r_[[[0]],theta[1:]])
    # print("INSIDE GRADIENT")
    # print('theta shape = ', theta.shape)
    # print('y shape = ', y.shape)
    # print('(h-y) shape = ', (h-y).shape)
    # print('grad shape = ', grad.shape)

    #### Must flatten the gradient from a multi dimensional array to a one dimensional array 
    return grad.flatten()

# n_labels = Number of different classes
# all_theta = each row of all_theta corresponds to the learned logistic regression parameters for one class
# features = x
# classes = y
def oneVsAll(features, classes, n_labels, reg):
    initial_theta = np.zeros((401,1))           ##401x1
    all_theta = np.zeros((n_labels, features.shape[1]))        ##10x401
    for i in range(1, n_labels+1):
        bool_matrix = (classes == i) & 1
        # print(features.shape)
        # print(classes.shape)
        res = minimize(costFunc, initial_theta, args=(reg, features, (bool_matrix)), method='Newton-CG', jac=gradient, options={'maxiter' : 15})
        all_theta[i-1] = res.x 

    return all_theta

#x.shape = (input_vector_size ,401)
#all_theta.shape = (10,401)
def predictor(x, all_theta):
    probability = sigmoid(x.dot(all_theta.T))           #size = (input_vector_size , 10)

    predict_vector = []
    for i in range(x.shape[0]):
        predict_vector.append(np.argmax(probability[i,:]))

    return(np.argmax(probability, axis=0)+1).reshape(-1,1)

if  __name__ == '__main__':
    data = loadmat('./ex3data1.mat')
    weights = loadmat('./ex3weights.mat')
    theta1, theta2  = weights['Theta1'], weights['Theta2']

    x, y   = data['X'], data['y']
    #Add the intercept term to the array
    x = np.concatenate((np.ones((5000,1)),x), axis=1)
    # theta = oneVsAll(x, y, 10, 0.1)
    # print(x.shape, y.shape, theta1.shape, theta2.shape)
    # pred = predictor(theta, x)
    # print('The accuracy of the training set is = ', 100*(y[y==pred]).size /x.shape[0])

    ############# SCI KIT LEARN LOGISTIC REGRESSION ##################

    # clf = LR(C=10, penalty='l2', solver='newton-cg', n_jobs=-1, warm_start=True, max_iter=100)
    # clf.fit(x[:,:], y.ravel())

    # pred2 = clf.predict(x[:,:]).reshape(-1,1)
    # print("The accuracy of trainig set is = ", 100*(y[y==pred2]).size / x.shape[0])

    ############ NEURAL NETWORK #######################


    data = loadmat('./ex3data1.mat')
    weights = loadmat('./ex3weights.mat')
    theta1, theta2  = weights['Theta1'], weights['Theta2']
    
    def predict_neural(t1, t2, x, y):
       z2 = t1.dot(x.T)
       a2 = np.concatenate((np.ones((x.shape[0],1 )), sigmoid(z2).T), axis=1)
       z3 = a2.dot(t2.T)
       a3 = sigmoid(z3)
       return (np.argmax(a3, axis=1) + 1)

    pred3 = predict_neural(theta1, theta2, x, y).reshape(-1,1)
    print("The accuracy of trainig set is = ", 100*(y[y==pred3]).size / x.shape[0])
