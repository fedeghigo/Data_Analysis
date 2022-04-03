## Main Class

from queue import Empty
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

class BayesGaussian(object):
    def __init__(self,df,  x_train, y_train, x_test, y_test) -> None:
        self.df=df
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def _calc_class_prior(self, test_value=1):
        outcome_count = sum(self.x_train.iloc[:,-1] == test_value)
        self.class_priors = outcome_count / len(self.y_train)
        return self.class_priors
    
    def subset(self):
        self.x_train_1=self.x_train[self.x_train["diagnosis"]==1]
        self.x_train_0=self.x_train[self.x_train["diagnosis"]==0]
        self.y_train_1=self.y_train[self.y_train["diagnosis"]==1]
        self.y_train_0=self.y_train[self.y_train["diagnosis"]==0]
        
        return self.x_train_1, self.x_train_0 , self.y_train_1 ,self.y_train_0

    def mean(self, df) :
        self.calculated_mean = []
        for i in self.df.columns:
            x_sum = sum(df[i])
            temp = x_sum / len(df[i])
            self.calculated_mean.append(temp)
        return self.calculated_mean

    def stdev(self,df ):
        self.calculated_std = []
        for i in self.x_train.columns:
            x_std = np.std(df[i])
            self.calculated_std.append(x_std)
        return self.calculated_std

    # Calculate the mean, stdev and count for each column in a dataset
    def summarize_dataset(self):
        return (
            print("Mean of the training  x", self.calculated_mean),
            print("Std of the training x", self.calculated_std),
            print("Len of the Training", len(self.x_train)),
        )

    def calculate_probability(self, x, mean, stdev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent
    
    
  
    def likelihood_list(self, df):
        self.likelihood_computed = pd.DataFrame()
        concat=pd.DataFrame() 
        mean_temp=self.mean(df)[:-1]
        stdev_temp=self.stdev(df)[:-1]
        #likeli=pd.DataFrame()  
        tt=pd.DataFrame()
        for i in range(len(df[:-1])):
            concat=pd.DataFrame()  
            for col in  range(len(df.columns)-1):                 
                temp = self.calculate_probability(
                df.iloc[i, col],
                mean_temp[col],
                stdev_temp[col],
                )            
                likeli =pd.Series(temp)
                concat = pd.concat([concat,likeli], axis=1)
            self.likelihood_computed= pd.concat([concat,self.likelihood_computed], axis=0)
        return self.likelihood_computed #self.likelihood_computed
    
    def single_likelihood(self, df ):
        mul_likelihood = self.likelihood_list(df).prod(axis=1) 
        return mul_likelihood
        
    def bayes_classification(self):
        self.y_computed = []
        likeli = np.array(self.single_likelihood(self.x_train_1))
        likeli_0 =np.array(self.single_likelihood(self.x_train_0))
        for i in range(len(self.x_test)-1):
            num=likeli[i] * self.class_priors
            self.den=likeli[i]*self.class_priors + likeli_0[i]*self._calc_class_prior(test_value=0) 
            bayes=num/self.den
            self.y_computed.append(bayes)
        return self.y_computed
    
    def normalize_data(self, threshold=0.5):
        self.pred=pd.Series(self.y_computed)
        self.pred[self.pred>=threshold]=1
        self.pred[self.pred<threshold]=0
        return self.pred
        
        
    def summary_bayes_classification(self,threshold=0.5):
        print("...... Mean .......")
        print("Train 1",self.mean(self.x_train_1) )
        print("Train 0",self.mean(self.x_train_0) )
        
        print("...... Std .......")
        print("Train 1",self.stdev(self.x_train_1) )
        print("Train 0",self.stdev(self.x_train_0) )
        
        print("...... Prior .......")
        print("Train 1",self._calc_class_prior(test_value=1))
        print("Train 0",self._calc_class_prior(test_value=0))  
        
        print("...... Likelihood .......")
        print("Train 1",self.single_likelihood(self.x_train_1 ))
        print("Train 0",self.single_likelihood(self.x_train_0))
        print("...... Normalization .......")
        print("Norm:",self.den)
        
        print("...... Result .......")
        print("Result:",self.y_computed)
        
        print("...... Metrics .......")
        print("Accuracy",accuracy_score(self.y_test["diagnosis"][:-1], self.normalize_data(threshold=threshold)))
        
        
