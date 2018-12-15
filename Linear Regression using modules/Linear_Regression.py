#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class  LReg:
    
    def __init__(self,X,y):
        m=len(y)
        n=np.shape(X)[1]
        self.X=X
        self.y=y
        self.w=np.zeros((n,1))
        self.c=[0]
        
    def Calcost(self):
        m=len(self.y)
        d=np.dot(self.X,self.w)+self.c-self.y
        J=(1/(2.0*m)) * np.sum(np.square(d))
        return J
    
    def gradient_descent(self,alpha,noi):
        m=len(self.y)
        for i in range(noi):
            pred=np.dot(self.X,self.w)+self.c
            w_grad=self.X.T.dot(pred-self.y)
            c_grad=np.sum(pred-self.y)
            self.w-=(alpha/m)*w_grad
            self.c-=(alpha/m)*c_grad
        return self.w,self.c
    
    def predict(self,X):
        y=X.dot(self.w)+self.c
        return y
    
    def accuracy(self,y_pred,y_test):
        error=(y_pred-y_test)/y_test *100
        acc=100-np.mean(error)
        return acc
    
  #_________________________________________________________________________#  

def normalize(X):
    mean=np.mean(X,axis=0)
    dev=np.std(X,axis=0)
    Xn=(X-mean)/dev
    return Xn,mean,dev



