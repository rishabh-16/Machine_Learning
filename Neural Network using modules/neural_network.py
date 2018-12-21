import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------MULTI CLASSIFIER CLASS----------------------------------------------------------------#

class nn:
    
    def fit(self,X,Y,hidden_layers_sizes):
        self.X=X
        self.Y=Y                                  #initialising everything
        n=self.X.shape[1]
        self.m=len(Y)
        
  #prints labels.....      
        self.labels=np.sort(np.unique(Y))
        print("labels are:",self.labels)       
        noc=len(self.labels)                      #number of classes
       
        """this block code helps to encode labels in Y (eg:[2,4]or["benign","malignant"]) into y (eg: [[0,1],[1,0]])"""
        
 #encoding...........
        self.y=np.zeros((self.m,noc))
        for i in range(self.m):
            self.y[i,np.argwhere(self.labels==Y[i])]=1
 #encoding done.........          
            
        h_sz=hidden_layers_sizes
        self.noh=len(h_sz)
        self.l_sz=np.insert(h_sz,[0,self.noh],[n,noc])         #it contains the size of each layer
                                              
        self.weights=[None]*(self.noh+1)                       #initialising weight list
        self.bias=[None]*(self.noh+1)                          #initialising bias list

 #giving random values to weights and bias..........     
        epsilon_init=0.12
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
    
    
    def shuffle_in_unison(self,a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
    
    
    
    def feedforward(self):
        self.a=[None]*(self.noh+2)  #initialises a list
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
        return self.weights,self.weights,self.bias,self.bias     #returning parameters
    
    
    
    def predict(self,X):                        #_________predicts the output________#
        a=[None]*(self.noh+2)
        a[0]=X
        for i in range(1,self.noh+2):
            a[i]=self.sigmoid(a[i-1].dot(self.weights[i-1])+self.bias[i-1])
            
        """maps each coded value to corresponding label eg. [0,1] to "benign" """
  #decodng....  
        p=a[self.noh+1]
        y_pred=self.labels[np.argmax(p,axis=1)]
  #done decoding.....

        return y_pred
    
    
    
    def accuracy(self,X_test,y_test):     #_________tests the accuracy of the model______#
        y_pred=self.predict(X_test)
        A=(y_pred==y_test)
        acc=np.mean(A)*100
        return acc
    
#----------------------------------------------------------------------------------------------------------------------------------------#        
    
    
        
    
    