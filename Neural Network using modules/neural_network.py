import numpy as np
import matplotlib.pyplot as plt

--------------------------------------------MULTI CLASSIFIER CLASS----------------------------------------------------------------

class nn:
    
    def fit(self,X,y,hidden_layers_sizes):
        self.X=X
        self.y=y                              #initialising everything
        self.m=y.shape[0]
        n=self.X.shape[1]
        noc=self.y.shape[1]
        h_sz=hidden_layers_sizes
        self.noh=len(h_sz)
        self.l_sz=np.insert(h_sz,[0,self.noh],[n,noc])         #it contains the size of each layer
        epsilon_init=0.12                                      
        self.weights=[None]*(self.noh+1)                       #initialising weight list
        self.bias=[None]*(self.noh+1)                          #initialising bias list
        for i in range(self.noh+1):
            self.weights[i]=np.random.rand(self.l_sz[i],self.l_sz[i+1]) * 2 * epsilon_init - epsilon_init
            self.bias[i]=np.random.rand(1,self.l_sz[i+1]) * 2 * epsilon_init - epsilon_init
            
    def normalize(self,X):                 #normalizing function
        mean=np.mean(X,axis=0)
        X=(X-mean)
        return X,mean
    
    def sigmoid(self,z):
        ans=1/(1+np.exp(-z))
        return ans   
    
    def feedforward(self):
        self.a=[None]*(self.noh+2)
        self.z=[0]*(self.noh+1)
        self.a[0]=self.X
        for i in range(1,self.noh+2):
            self.a[i]=self.sigmoid(self.a[i-1].dot(self.weights[i-1])+self.bias[i-1])
              
    def backprop(self):
        d=[None]*(self.noh+2)  
        d[self.noh+1]=self.a[self.noh+1]-self.y
        for i in range(self.noh,0,-1):
            d[i]=d[i+1].dot(self.weights[i].T)*self.a[i]*(1-self.a[i])
        weights_grad=[None]*(self.noh+1)
        bias_grad=[None]*(self.noh+1)
        for i in range(self.noh+1):
            weights_grad[i]=(1/self.m)*np.dot(self.a[i].T,d[i+1])+(self.LAMBDA/self.m)*self.weights[i]    #calculating gradients
            bias_grad[i]=(1/self.m)*np.sum(d[i+1],axis=0)
            self.weights[i]-=self.alpha*weights_grad[i]
            self.bias[i]-=self.alpha*bias_grad[i]        
    
    def backprop_mini(self,b_sz):
        for j in range(0,self.m,b_sz):                #---to avoid overflow---#
            if(j+b_sz<self.m):
                e=j+b_sz
            else:
                e=self.m
            d=[None]*(self.noh+2)  
            d[self.noh+1]=(self.a[self.noh+1])[j:e,:]-self.y[j:e,:]
            for i in range(self.noh,0,-1):
                d[i]=d[i+1].dot(self.weights[i].T)*self.a[i][j:e,:]*(1-self.a[i][j:e,:])
            weights_grad=[None]*(self.noh+1)
            bias_grad=[None]*(self.noh+1)
            for i in range(self.noh+1):
                weights_grad[i]=(1/b_sz)*np.dot(self.a[i][j:e,:].T,d[i+1])+(self.LAMBDA/b_sz)*self.weights[i]
                bias_grad[i]=(1/b_sz)*np.sum(d[i+1],axis=0)
                self.weights[i]-=self.alpha*weights_grad[i]
                self.bias[i]-=self.alpha*bias_grad[i]
            self.feedforward()                                                  
                                   
    def gradient_descent(self,alpha,noi,LAMBDA=0):
        self.LAMBDA=LAMBDA
        self.alpha=alpha
        for i in range(noi):
            self.feedforward()
            self.backprop()
        return self.weights,self.weights,self.bias,self.bias    #returning parameters
    
    
    def mini_batch_gradient_descent(self,alpha,noi,b_sz,LAMBDA=0):
        self.LAMBDA=LAMBDA
        self.alpha=alpha
        for i in range(noi):
            self.feedforward()
            self.backprop_mini(b_sz)
        return self.weights,self.weights,self.bias,self.bias
    
    
    def predict(self,X):                        #_________predicts the output________#
        a=[None]*(self.noh+2)
        a[0]=X
        for i in range(1,self.noh+2):
            a[i]=self.sigmoid(a[i-1].dot(self.weights[i-1])+self.bias[i-1])
        y=a[self.noh+1]
        p = np.argmax(y, axis=1)
        y=np.zeros(y.shape)
        for i in range(len(p)):
            y[i,p[i]]=1
        return y
    
    def accuracy(self,X_test,y_test):     #_________tests the accuracy of the model______#
        y_pred=self.predict(X_test)
        p1 = np.argmax(y_pred, axis=1)
        p2 = np.argmax(y_test, axis=1)
        A=(p1==p2)
        print(A)
        acc=np.mean(A)*100
        return acc
    
------------------------------------------------------------------------------------------------------------------------------------------        
    
    
        
    
    