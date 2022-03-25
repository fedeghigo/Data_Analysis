## Main Class
from re import X
from tkinter import Y
from typing import List
import pandas as pd
import numpy as np


class NaiveBayes:
    def __init__(self, features: pd.DataFrame, label: pd.Series) -> None:
        self.features = features
        self.label = label
        # self.prior=prior

    def likelihood(self):
        pass

    def prior(self):
        _num = len(self.label[self.label == 1])
        _den = len(self.label)

    def normalization(self):
        pass

    def Bayes(self):
        pass

    def IterateBayes(self):
        pass


# nb =NaiveBayes()

