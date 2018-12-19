import numpy as np


class KNeighborsClassifier:
    def fit(self,X,y,k=-1):
        self.X=X
        self.y=y
        n_c=len(np.unique(y))
        if(k==-1):
            self.k=n_c+1
        else:
            self.k=k
            
    def normalize(self,X):
        mean=np.mean(X,axis=0)
        X=X-mean
        return X
   
    def k_neighbors(self,dis):
        neighbors=self.y[np.argsort(dis)]
        knn=neighbors[:self.k]
        return knn
    
    def max_occurance(self,kn):
        counts = np.bincount(kn.astype(int))
        c=np.argmax(counts)
        return c
    
    def det_class(self,el):
        sq_dis=np.sum(np.square(self.X-el),axis=1)
        kn=self.k_neighbors(sq_dis)
        c=self.max_occurance(kn)
        return c
    
    def predict(self,X):
        m=X.shape[0]
        y_pred=np.zeros(m)
        for i in range(m):
            y_pred[i]=self.det_class(X[i,:])
        return y_pred
     
    def accuracy(self,X_test,y_test):
        y_pred=self.predict(X_test)
        a=y_pred==y_test
        acc=np.mean(a)*100
        return acc
    