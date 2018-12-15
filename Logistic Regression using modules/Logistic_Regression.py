#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np

def sigmoid(z):
    ans=1/(1+np.exp(-z))
    return ans

#_______________________________________________________________________________________#
class  LogReg:
    
    def __init__(self,X,y):
        m=len(y)
        n=np.shape(X)[1]
        self.X=np.hstack((np.ones((m,1)),X))
        self.y=y
        self.theta=np.zeros((n+1,1))
        
    def Calcost(self,theta):
        m=len(self.y)
        pred=sigmoid(self.X.dot(theta))
        J=(1/m) * (-self.y.T.dot(np.log(pred))-(1-self.y).T.dot(np.log(1-pred)))
        return J
    
    def gradient_descent(self,alpha,noi):
        m=len(self.y)
        for i in range(noi):
            pred=sigmoid(self.X.dot(self.theta))
            self.theta = self.theta - (alpha/m)*(self.X.T.dot(pred-self.y))
        return self.theta
    
    def predict(self,X):
        m=np.shape(X)[0]
        X=np.hstack((np.ones((m,1)),X))
        y=sigmoid(X.dot(self.theta))>=0.5
        return y.astype(int)
#________________________________________________________________________________________#

def normalize(X):
    mean=np.mean(X,axis=0)
    dev=np.std(X,axis=0)
    Xn=(X-mean)/dev
    return Xn,mean,dev

def accuracy(y_pred,y_test):
    A=(y_pred==y_test)
    acc=np.mean(A)*100
    return acc
    


