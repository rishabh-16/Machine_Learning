import numpy as np

"""=====================================MODULE FOR IMPLEMENTING KNN CLASSIFICATION======================================================"""

class KNeighborsClassifier:
    
    def fit(self,X,y,k=-1):
        self.X=X
        self.y=y
        labels=np.unique(y)
        print("labels:",labels)   #prints labels or class codes.......
        n_c=len(labels)
        """
        if the value of k is not given by the user then it assigns
        it value as one more than number of classes ie. n_c 
        """
        if(k==-1):
            self.k=n_c+1
        else:
            self.k=k
            
            
    def shuffle_in_unison(self,a, b):
        """
        this function simply takes two arrays and shuffle them
        in such a way that their corressponding values remain same
        """
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        
            
    def normalize(self,X):
        mean=np.mean(X,axis=0)
        X=X-mean
        return X
   
    def k_neighbors(self,dis):
        """
        it takes the distance array and 
        returns the k nearest neighbouring classes
        """
        neighbors=self.y[np.argsort(dis)]
        knn=neighbors[:self.k]
        return knn
    
    def max_occurance(self,kn):
        """
        this function takes the k nearest neighbours
        and returns the class with maximum occurance
        """
        counts = np.bincount(kn.astype(int))
        c=np.argmax(counts)
        return c
    
    def det_class(self,el):
        """
        this function takes an datapoint and determines 
        the class in which it is most likely to fall into
        """
        sq_dis=np.sum(np.square(self.X-el),axis=1)
        kn=self.k_neighbors(sq_dis)
        c=self.max_occurance(kn)
        return c
    
    def predict(self,X):                           #_________predicts the output________#
        m=X.shape[0]
        y_pred=np.zeros(m)
        for i in range(m):
            y_pred[i]=self.det_class(X[i,:])
        return y_pred
     
    def accuracy(self,X_test,y_test):         #_________tests accuracy of the model______#
        y_pred=self.predict(X_test)
        a=y_pred==y_test
        acc=np.mean(a)*100
        return acc
    
    """==================================================XXX======================================================================="""

    
    