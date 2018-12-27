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
        print(self.l_sz)
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
        """
        it is the activation function for this neural network module
        it takes an array and return the activated array
        """
        ans=1/(1+np.exp(-z))
        return ans   
    
    
    def shuffle_in_unison(self,a, b):
        """
        this function simply takes two arrays and shuffle them
        in such a way that their corressponding values remain same
        """
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
    
    
    
    def feedforward(self):
        """
        this function calculates the value of each layer using current weights and bias
        a contains the value of each layer
        """
        self.a=[None]*(self.noh+2)  #initialises a list
        self.a[0]=self.X
        for i in range(1,self.noh+2):
            self.a[i]=self.sigmoid(self.a[i-1].dot(self.weights[i-1])+self.bias[i-1])
            #self.a[i]=self.normalize(np.array(self.a[i]))
            
            
              
    def backprop(self):
        
        """
        this function backpropagate error in output layer to get error in hidden layers and further calculates the gradient
        d contains the error in each layer
        """
        
        d=[None]*(self.noh+2)  
        d[self.noh+1]=self.a[self.noh+1]-self.y
        
        for i in range(self.noh,0,-1):
            d[i]=d[i+1].dot(self.weights[i].T)*self.a[i]*(1-self.a[i])     #....calculates error in hidden layers
            
        weights_grad=[None]*(self.noh+1)
        bias_grad=[None]*(self.noh+1)
        for i in range(self.noh+1):
            weights_grad[i]=(1/self.m)*np.dot(self.a[i].T,d[i+1])+(self.LAMBDA/self.m)*self.weights[i]    #calculating gradients
            bias_grad[i]=(1/self.m)*np.sum(d[i+1],axis=0)
            self.weights[i]-=self.alpha*weights_grad[i]
            self.bias[i]-=self.alpha*bias_grad[i]        
            
            
    
    def backprop_mini(self,b_sz):
        """
        the value of e takes care of overflow
        for example:
            if m = 10
            and b_sz = 3
            then correspond mini batch sizes will be : {3,3,3,1}
            thus avoiding error
        """
        for j in range(0,self.m,b_sz):                #---to avoid overflow---#
            if(j+b_sz<self.m):
                e=j+b_sz
            else:
                e=self.m
                
            d=[None]*(self.noh+2)  
            d[self.noh+1]=(self.a[self.noh+1])[j:e,:]-self.y[j:e,:]
            for i in range(self.noh,0,-1):
                d[i]=d[i+1].dot(self.weights[i].T)*self.a[i][j:e,:]*(1-self.a[i][j:e,:])  #....calculates error in hidden layers
                
            weights_grad=[None]*(self.noh+1)       #.....calculates gradients
            bias_grad=[None]*(self.noh+1)
            for i in range(self.noh+1):
                weights_grad[i]=(1/b_sz)*np.dot(self.a[i][j:e,:].T,d[i+1])+(self.LAMBDA/b_sz)*self.weights[i]
                bias_grad[i]=(1/b_sz)*np.sum(d[i+1],axis=0)
                self.weights[i]-=self.alpha*weights_grad[i]                                
                self.bias[i]-=self.alpha*bias_grad[i]
            self.feedforward()   
            
    def Calcost(self):        #______function to calculate cost______#
        """
        it is used to calculate cost of the model 
        at the current value of weights and bias,
        the value of cost should be low for a effective model
        """
        j=0
        for i in range(self.m):
            j += np.log(self.a[self.noh+1][i, ]).dot(-self.y[i, ].T) - np.log(1 - self.a[self.noh+1][i, ]).dot(1 - self.y[i, ].T)
        j /= self.m
        return j


      
                                   
    def gradient_descent(self,alpha,noi,LAMBDA=0):
        self.LAMBDA=LAMBDA
        self.alpha=alpha
        self.noi=noi
        self.Jv=np.zeros(noi)
        for i in range(noi):
            self.feedforward()
            self.backprop()
            self.Jv[i]=self.Calcost()
        return self.weights,self.weights,self.bias,self.bias    #returning parameters
    
    
    
    
    def mini_batch_gradient_descent(self,alpha,noi,b_sz,LAMBDA=0):
        self.LAMBDA=LAMBDA
        self.alpha=alpha
        self.Jv=np.zeros(noi)
        self.noi=noi
        for i in range(noi):
            self.feedforward()
            self.backprop_mini(b_sz)
            self.Jv[i]=self.Calcost()
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
       
    
#----------------------------------------------------------------------------------------------------------------------------------------#        
    
    
        
    
    