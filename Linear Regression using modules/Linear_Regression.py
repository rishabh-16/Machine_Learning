#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class  LReg:
    
    def __init__(self,X,y):
        m=len(y)
        n=np.shape(X)[1]
        self.X=np.hstack((np.ones((m,1)),X))
        self.y=y
        self.theta=np.zeros((n+1,1))
        
    def Calcost(self,theta):
        m=len(self.y)
        d=self.X.dot(theta)-self.y
        J=(1/(2.0*m)) * np.sum(np.square(d))
        return J
    
    def gradient_descent(self,alpha,noi):
        m=len(self.y)
        for i in range(noi):
            d=self.X.dot(self.theta)-self.y
            self.theta = self.theta - (alpha/m)*(self.X.T.dot(d))
        return self.theta
    
    def predict(self,X):
        m=np.shape(X)[0]
        X=np.hstack((np.ones((m,1)),X))
        y=X.dot(self.theta)
        return y
  #_________________________________________________________________________#  

def normalize(X):
    mean=np.mean(X,axis=0)
    dev=np.std(X,axis=0)
    Xn=(X-mean)/dev
    return Xn,mean,dev

def accuracy(y_pred,y_test):
    error=(y_pred-y_test)/y_test *100
    acc=100-np.mean(error)
    return acc
    

