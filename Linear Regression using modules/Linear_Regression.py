#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class  LReg:
    
    def fit(self,X,y,LAMBDA=0):        
        m=len(y)
        n=np.shape(X)[1]
        self.X=X
        self.y=y
        self.w=np.random.random((n,1)) #weight
        self.b=[0]  #bias
        self.LAMBDA=LAMBDA #---regularization factor---#
        
        
    def normalize(self,X):
        mean=np.mean(X,axis=0)
        dev=np.std(X,axis=0)
        Xn=(X-mean)/dev
        return Xn,mean,dev
        
    def Calcost(self):                      #______function to calculate cost______#
        m=len(self.y)
        d=np.dot(self.X,self.w)+self.b-self.y
        reg=(self.LAMBDA/(2.0*m))*np.sum(np.square(self.w))   
        J=(1/(2.0*m)) * np.sum(np.square(d))+reg
        return J
    
    def gradient_descent(self,alpha,noi):         #______optimizes w,b_______#
        m=len(self.y)
        self.Jv=np.zeros(noi)
        for i in range(noi):
            pred=np.dot(self.X,self.w)+self.b
            w_grad=self.X.T.dot(pred-self.y) + (self.LAMBDA/m)*self.w
            b_grad=np.sum(pred-self.y)
            self.w-=(alpha/m)*w_grad
            self.b-=(alpha/m)*b_grad
            self.Jv[i]=self.Calcost()
        return self.w,self.b
    
    def mini_batch_gradient_descent(self,sz,alpha,noi):   
        m=len(self.y)
        np.random.shuffle([self.X,self.y])           #----shuffles data-----#
        for i in range(noi):
            for j in range(0,m,sz):            
                if(j+sz<m):               #---to avoid overflow---#
                    e=j+sz       
                else:
                    e=m
                X_batch=self.X[j:e,:]
                y_batch=self.y[j:e,:]
                pred=np.dot(X_batch,self.w)+self.b
                w_grad=X_batch.T.dot(pred-y_batch) + (self.LAMBDA/sz)*self.w
                b_grad=np.sum(pred-y_batch)
                self.w-=(alpha/sz)*w_grad
                self.b-=(alpha/sz)*b_grad
        return self.w,self.b    
    
    def predict(self,X):                   #_________predicts the output________#
        y=X.dot(self.w)+self.b
        return y
    
    def accuracy(self,X_test,y_test):      #_________tests accuracy of the model______#
        y_pred=self.predict(X_test)
        error=(y_pred-y_test)/y_test *100
        acc=100-abs(np.mean(error))
        return acc
    
  #________________________________________________________________________________#  



