import numpy as np

"""=====================================MODULE FOR IMPLEMENTING DECISION TREE CLASSIFICATION=========================================="""

class Decision_Tree:
    def fit(self,X,y):
        try:
            self.X=X.tolist()
            self.y=y.tolist()    #THIS CONVERTS X AND y TO LIST IF IT  IS NOT
        except:
            self.X=X
            self.y=y
        self.m=len(X)
        self.n=len(X[0])
        self.Node=self.build_tree(self.X,self.y)      #THIS NODE IS ROOT NODE OF THE TREE
        
        
    def shuffle_in_unison(self,a, b):
        """
        this function simply takes two arrays and shuffle them
        in such a way that their corressponding values remain same
        """
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        
   
    def class_counts(self,y):
        """
        THIS FUNCTION COUNTS THE OCCURANCES OF LABELS 
        AND RETURNS THE DICTIONARY
        """
        counts={}
        for label in y:
            if label not in counts:
                counts[label]=0
            counts[label]+=1
        return counts           
    
    def gini(self,X,y):
        """
        THIS FUNCTION RETURNS THE GINI IMPURITY IN THE GIVEN BRANCH OF TREE
        """
        counts = self.class_counts(y)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(X))
            impurity -= prob_of_lbl**2
        return impurity
    
    
    def info_gain(self, true_X, true_y, false_X, false_y, current_uncertainty):
        """
        IT RETURNS THE AMOUNT OF INFORMATION GAINED ACROSS A NODE
        """
        p = float(len(true_X)) / (len(true_X) + len(false_X))
        return current_uncertainty - p * self.gini(true_X, true_y) - (1 - p) * self.gini(false_X, false_y)



    class Question:
        def __init__(self,column,value):
            self.column=column
            self.value=value
            
        def match(self,example):
            val=example[self.column]
            if isinstance(val,int) or isinstance(val,float):
                return val>=self.value
            else:
                return val==self.value
            

    def partition(self,X,y,question):
        """
        THIS FUNCTION PARTITIONS THE DATA INTO TWO BRANCHS ACCORDING TO A GIVEN CONDITION
        AND RETURNS THE BRANCHES
        """
        true_X,true_y,false_X,false_y=[],[],[],[]
        for i in range(len(X)):
            if question.match(X[i]):
                true_X.append(X[i])
                true_y.append(y[i])
            else:
                false_X.append(X[i])
                false_y.append(y[i])
        return true_X,true_y,false_X,false_y
    
    



    def find_best_split(self,X,y):
        """
        IT FINDS THE BEST QUESTION TO BE ASKED TO HAVE MAXIMUM INFORMATION GAIN
        """
        best_gain = 0  
        best_question = None  
        current_uncertainty = self.gini(X,y)  
        for col in range(self.n):  
            values = set([row[col] for row in X])  
            for val in values:  
                question = self.Question(col, val)
                true_X, true_y, false_X, false_y = self.partition(X, y, question)
                if (len(true_X) == 0 or len(false_X) == 0):
                    continue
                gain = self.info_gain(true_X, true_y, false_X, false_y, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question
    
    
    class Leaf:
        """
        IT IS THE LEAF NODE THAT CONTAINS THE MOST CLASSIFIED INFO
        """
        def __init__(self,X,y):
            counts=Decision_Tree().class_counts(y)
            total=sum(counts.values())
            for label in counts.keys():
                counts[label]=str(counts[label]/total * 100)+"%"
            self.predictions=counts
        
        
    class Decision_Node:
        """
        IT IS THE NODE FROM WHICH BRANCHING OCCURS
        """
        def __init__(self,question,true_branch,false_branch):
            self.true_branch=true_branch
            self.false_branch=false_branch
            self.question=question
    
    def build_tree(self, X, y):
        """
        THIS FUNCTIONS DO THE BRANCHING RECURSIVELY AND RETURNS THE RESPECTIVE NODES
        """
        gain, question=self.find_best_split(X,y)
        if gain == 0:
            return self.Leaf(X,y)
        true_X,true_y,false_X,false_y=self.partition(X,y,question)
        true_branch=self.build_tree(true_X,true_y)
        false_branch=self.build_tree(false_X,false_y)
        return self.Decision_Node(question,true_branch,false_branch)
    
            
            
    def classify(self,Node,example):
        """
        IT IS USED TO CLASSIFY AN EXAMPLE BY USIND THE TREE
        """
        if isinstance(Node,self.Leaf):
            return Node.predictions
        else:
            if(Node.question.match(example)):
                return self.classify(Node.true_branch,example)
            else:
                return self.classify(Node.false_branch,example) 
            
    def predict(self,X_test):    #_________PREDICTS THE OUTPUT________#
        y_pred=[]
        for example in X_test:
            d=self.classify(self.Node,example)
            v=list(d.values())
            k=list(d.keys())
            y_pred.append(k[v.index(max(v))])
        return np.array(y_pred)
    
    def accuracy(self,X_test,y_test):         #_________TESTS THE ACCURACY OF THE MODEL______#
        y_pred=self.predict(X_test)
        a=np.array(y_pred==y_test)
        acc=np.mean(a)*100
        return acc
                 
     
    def predict_prob(self,X_test):          #__________PREDICTS THE PROBABILITY________#
        y_pred=[]
        for example in X_test:
            y_pred.append(self.classify(self.Node,example))
        return y_pred
        
        
            
"""==================================================XXX======================================================================="""
        
        
