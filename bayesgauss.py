## Main Class

from queue import Empty
from typing import List
import pandas as pd
import numpy as np


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
        mul_likelihood = df.prod(axis=1) 
        return mul_likelihood
        
    def bayes_classification(self):
        y_computed = []
        likeli = np.array(self.single_likelihood(self.x_train_1))
        likeli_0 =np.array(self.single_likelihood(self.x_train_0))
        for i in range(len(self.x_test)-1):
            num=likeli[i] * self.class_priors
            den=likeli[i]*self.class_priors + likeli_0[i]*self._calc_class_prior(test_value=0) 
            bayes=num/den
            y_computed.append(bayes)
        return y_computed
        
        
    def bayes(self, columns=0):
        y_computed = []
        for i in range(len(self.x_test)):
            num = self.likelihood_computed[i] * self.class_priors
            den = sum(self.likelihood_computed) + self.class_priors
            bayes = num / den
            y_computed.append(bayes)
        return y_computed

    # def likelihood(self):
    #     self.likelihood_list = []
    #     for i in self.x_train.iloc[:, 0]:
    #         likelihood = self.calculate_probability(
    #             i, self.calculated_mean[0], self.calculated_std[0]
    #         )
    #         self.likelihood_list.append(likelihood)
    #     return self.likelihood_list

    # def bayes(self, df, priors, likelihood):
    #     for i in df:
    #         bayes = priors * likelihood / sum(priors * likelihood)
    #     return bayes

    # def gaussian(self, x: list):
    #     t = pd.DataFrame()
    #     for i in range(len(self.x_train.columns) - 1):
    #         exponent = np.exp(-((x - self.mean()) ** 2 / (2 * self.stdev() ** 2)))
    #         norm = 1 / (np.sqrt(2 * np.pi) * self.stdev()) * exponent
    #     return norm

    # def gaussian( x , self.mean(), self.stdev() ):

    #     for i in len(self.x_train.columns):
    #         exponent = np.exp(-((x[i]-self.mean())[i]**2 / (2 * self.stdev()[i]**2 )))
    #     return (1 / (sqrt(2 * np.pi) * self.stdev()[i])) * exponent


#     def gauss_dist(self):
#         gaussian = 1 / (
#             self.devstd
#             * np.sqrt(2 * np.pi)
#             * np.exp(-1 / 2 * (self.x - self.mu / self.devstd))
#         )
#         return gaussian

#     def norm_gauss(self):
#         z = c - mu / devstd
#         norm = 1 / (np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * z ** 2)
#         return norm

#     def norm_gauss_complementare(self):
#         z = c - mu / devstd
#         norm = 1 / (np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * z ** 2)
#         return 1 - norm
#     def prob(self):
#         z = c - mu / devstd
#         return z


# class NaiveBayes:
#     def __init__(self, features: pd.DataFrame, label: pd.Series) -> None:
#         self.features = features
#         self.label = label
#         # self.prior=prior

#     def likelihood(self):
#         pass

#     def prior(self):
#         _num = len(self.label[self.label == 1])
#         _den = len(self.label)

#     def normalization(self):
#         pass

#     def Bayes(self):
#         pass

#     def IterateBayes(self):
#         pass


# # nb =NaiveBayes()

#f
