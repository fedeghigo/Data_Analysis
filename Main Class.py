## Main Class

from typing import List
import pandas as pd
import numpy as np


class Gaussian:
    def __init__(self, mu) -> None:
        self.mu = mu
        self.devstd = devstd
        self.x = x

    def gauss_dist(self):
        gaussian = 1 / (
            self.devstd
            * np.sqrt(2 * np.pi)
            * np.exp(-1 / 2 * (self.x - self.mu / self.devstd))
        )
        return gaussian

    def norm_gauss(self):
        z = c - mu / devstd
        norm = 1 / (np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * z ** 2)
        return norm

    def norm_gauss_complementare(self):
        z = c - mu / devstd
        norm = 1 / (np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * z ** 2)
        return 1 - norm
    def prob(self):
        z = c - mu / devstd
        return z


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

