#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt


"""=====================================MODULE FOR IMPLEMENTING LINEAR REGRESSION======================================================"""

class  LogReg:
    
    def fit(self,X,y,LAMBDA=0):
        m=len(y)
        n=np.shape(X)[1]
        self.X=X
        self.y=y.reshape(len(y),1)
        self.w=np.random.rand(n,1) #weight
        self.b=np.random.rand(1)    #bias
        self.LAMBDA=LAMBDA     #---regularization factor---#
        
    def sigmoid(self,z):
        ans=1/(1+np.exp(-z))
        return ans
    
    def normalize(self,X):
        mean=np.mean(X,axis=0)
        dev=np.std(X,axis=0)
        Xn=(X-mean)/dev
        return Xn,mean,dev
    
    
    def shuffle_in_unison(self,a, b):
        """
        this function simply takes two arrays and shuffle them
        in such a way that their corressponding values remain same
        """
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        

        
    def Calcost(self):        #______function to calculate cost______#
        """
        it is used to calculate cost of the model 
        at the current value of weights and bias,
        the value of cost should be low for a effective model
        """
        m=len(self.y)
        pred=self.sigmoid(np.dot(self.X,self.w)+self.b)
        reg=(self.LAMBDA/(2.0*m))*np.sum(np.square(self.w))
        J=(1/m) * (-self.y.T.dot(np.log(pred))-(1-self.y).T.dot(np.log(1-pred))) + reg
        return J
    
    
    def gradient_descent(self,alpha,noi):            #______optimizes w,b_______#
        m=len(self.y)
        self.noi=noi
        self.Jv=np.zeros(noi)
        for i in range(noi):
            pred=self.sigmoid(np.dot(self.X,self.w)+self.b)
            w_grad=self.X.T.dot(pred-self.y) + (self.LAMBDA/m)*self.w
            b_grad=np.sum(pred-self.y)
            self.w-=(alpha/m)*w_grad 
            self.b-=(alpha/m)*b_grad
            self.Jv[i]=self.Calcost()
        return self.w,self.b
    
    def mini_batch_gradient_descent(self,sz,alpha,noi):
        """
        it is effective for huge data, 
        it divides the data into minibatches and operates on that
        thus compromising a bit on accuracy but takes less time to optimize
        """
        m=len(self.y)
        self.noi=noi
        self.Jv=np.zeros(noi)
        np.random.shuffle([self.X,self.y])             #----shuffles data-----#
        for i in range(noi):
            for j in range(0,m,sz):                #---to avoid overflow---#
                if(j+sz<m):
                    e=j+sz
                else:
                    e=m
                X_batch=self.X[j:e,:]
                y_batch=self.y[j:e,:]
                pred=self.sigmoid(np.dot(X_batch,self.w)+self.b)
                w_grad=X_batch.T.dot(pred-y_batch) + (self.LAMBDA/sz)*self.w
                b_grad=np.sum(pred-y_batch)
                self.w-=(alpha/sz)*w_grad
                self.b-=(alpha/sz)*b_grad
            self.Jv[i]=self.Calcost()
        return self.w,self.b      
    
    def predict(self,X):                        #_________predicts the output________#
        y=self.sigmoid(X.dot(self.w)+self.b)>=0.5
        return y.astype(int)
    
    def accuracy(self,X_test,y_test):     #_________tests accuracy of the model______#
        y_pred=self.predict(X_test)
        y_test=y_test.reshape((len(y_test),1))
        A=(y_pred==y_test)
        acc=np.mean(A)*100
        return acc
    
    def plot_learning_curve(self):
        """
        it plots the cost vs number of iterations curve to 
        help the user decide the number of iterations and alpha value
        correspondingly
        """
        plt.figure()
        plt.plot(range(self.noi),self.Jv)
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost')
        plt.title('Cost vs Number of iterations curve')

        
    """==================================================XXX======================================================================="""




    


