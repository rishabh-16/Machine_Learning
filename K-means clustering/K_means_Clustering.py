import numpy as np

"""=====================================MODULE FOR IMPLEMENTING K MEANS CLUSTERING======================================================"""

class KMeans:
    
    def fit(self,X):
        self.X=X
        self.m=X.shape[0]
             
       
    def normalize(self,X):
        mean=np.mean(X,axis=0)
        X=(X-mean)/np.std(X,axis=0)
        return X
   
   
    def random_init(self):
        """
        this function initialises centroids with randomly 
        picked datapoints
        """
        indices=np.random.choice(self.m,self.k)
        self.centroids=self.X[indices,:]
    
    
    def predict_clusters(self,X):
        """
        this function takes array of datapoints and returns the
        corresponding array of the cluster label to which it belongs
        """
        c=np.argmin(np.sum((self.centroids-X[:,None,:])**2,axis=2),axis=1)
        return c
          
    
    def compute_means(self):
        """
        this functions move the centroids to the means of its corresponding
        datasets position
        """
        self.centroids=[self.X[np.argwhere(self.c==i).flatten(),:].mean(0) for i in range(self.k)]
            
            
    def assign_clusters(self,noi,n_clusters):
        self.k=n_clusters
        self.c_labels=np.arange(self.k)
        print("Cluster Labels are: ",self.c_labels)
        self.random_init()
        for i in range(noi):
            self.c=self.predict_clusters(self.X)
            self.compute_means()
        return self.c.astype(int)
    
 
       
     
    """==================================================XXX======================================================================="""

    
    