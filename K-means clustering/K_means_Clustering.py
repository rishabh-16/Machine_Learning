import numpy as np

"""=====================================MODULE FOR IMPLEMENTING K MEANS CLUSTERING======================================================"""

class Kclassifier:
    
    def fit(self,X,k):
        self.X=X
        self.k=k
        self.m=X.shape[0]
        self.random_initialize()    
            
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
   
   
    def random_initialize(self):
        indices=np.random.choice(self.m,self.k)
        self.centroids=self.X[indices,:]
    
    
    def find_closest_cetroids(self):
        self.c=np.zeros(self.m)
        for i in range(self.m):
            self.c[i]=np.argmax(np.sum((self.centroids-self.X[i,:])**2,axis=1))
    
    def compute_means(self):
        for i in range(self.k):
            self.centroids[i]=self.X[np.ravel(np.where(self.c==i)).astype('int'),:].mean(0)
            
    def get_clusters(self,noi):
        for i in range(noi):
            self.find_closest_centroids()
            self.compute_means()
        return c
    """==================================================XXX======================================================================="""

    
    