#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np


#_______________________________________________________________________________________#
class  LogReg:
    
    def __init__(self,X,y):
        m=len(y)
        n=np.shape(X)[1]
        self.X=X
        self.y=y
        self.w=np.zeros((n,1))
        self.c=[0]
        
    def sigmoid(self,z):
        ans=1/(1+np.exp(-z))
        return ans

        
    def Calcost(self):
        m=len(self.y)
        pred=self.sigmoid(np.dot(self.X,self.w)+self.c)
        J=(1/m) * (-self.y.T.dot(np.log(pred))-(1-self.y).T.dot(np.log(1-pred)))
        return J
    
    def gradient_descent(self,alpha,noi):
        m=len(self.y)
        for i in range(noi):
            pred=self.sigmoid(np.dot(self.X,self.w)+self.c)
            w_grad=self.X.T.dot(pred-self.y)
            c_grad=np.sum(pred-self.y)
            self.w-=(alpha/m)*w_grad
            self.c-=(alpha/m)*c_grad
        return self.w,self.c
    
    def predict(self,X):
        y=self.sigmoid(X.dot(self.w)+self.c)>=0.5
        return y.astype(int)
    
    def accuracy(self,y_pred,y_test):
        A=(y_pred==y_test)
        acc=np.mean(A)*100
        return acc
#________________________________________________________________________________________#

def normalize(X):
    mean=np.mean(X,axis=0)
    dev=np.std(X,axis=0)
    Xn=(X-mean)/dev
    return Xn,mean,dev


    


