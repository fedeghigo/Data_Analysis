##print("hello")
# %load '/Users/mari/Master/teaching/Py4Master/my-BayesCancerTest.py'
# importing necessary libraries
import sys
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

#    +++++++++++++++++++++++++
# DATA

print("+-------------------------------+")
print("| Bayes Unfolding: reading data |")
print("+-------------------------------+")
print()

#
# https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

cancer = load_breast_cancer()
# data = pd.DataFrame(data=cancer.data, columns=cancer.feature_names) # data from library
data = pd.read_csv(
    r"G:\Il mio Drive\1Master Data Analytics and data science\lezioni_MDA\data analysis\codice\dp-export-414127.csv"
)  # data from local file, only 4 columns
print("data: ")
print(data)
print("columns: ")
print(data.columns)
data["diagnosis"] = cancer.target
data = data[
    ["mean radius", "mean texture", "mean smoothness", "diagnosis"]
]  # select these 4 columns
print("data selection \n", data)
print("element 2 2: ", data.iloc[2, 2])  # single element
s1data = data["mean texture"]
print("column mean texture \n", s1data)
print("element 2 of mean texture \n", s1data.iloc[2])
data.head(10)  # select first 10 rows
# DATA
print("+---------------------------------------------+")
print("| Bayes Unfolding: data splitting and slicing |")
print("+---------------------------------------------+")
print()
print()

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(data, test_size=0.2, random_state=41)
print("df train")
print(df_train)
print("df test")
print(df_test)
nbins = 15

# slicing
X_test = df_test.iloc[
    :, :-1
].values  # from test DataFrame: all rows & all columns except last
D_test = df_test.iloc[:, -1].values  # from test DataFrame: all rows & last column
X1_test = df_test.iloc[:, 0].values  # from test DataFrame: all rows & first column
X2_test = df_test.iloc[:, 1].values  # from test DataFrame: all rows & second column
print("Xtest ", len(X_test))
print(X_test)
print("Ytest ", len(D_test))
print(D_test)
print("Z1test ", len(X1_test))
print(X1_test)
print("Z2test ", len(X2_test))
print(X2_test)

X_train = df_train.iloc[:, :-1].values  # from train DataFrame ..........
D_train = df_train.iloc[:, -1].values
X1_train = df_train.iloc[:, 0].values
X2_train = df_train.iloc[:, 1].values


# DATA
print("+---------------------------------------------+")
print("| Bayes Unfolding: data splitting and slicing |")
print("+---------------------------------------------+")
print()
print()

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(data, test_size=0.2, random_state=41)
print("df train")
print(df_train)
print("df test")
print(df_test)
nbins = 15

# slicing
X_test = df_test.iloc[
    :, :-1
].values  # from test DataFrame: all rows & all columns except last
D_test = df_test.iloc[:, -1].values  # from test DataFrame: all rows & last column
X1_test = df_test.iloc[:, 0].values  # from test DataFrame: all rows & first column
X2_test = df_test.iloc[:, 1].values  # from test DataFrame: all rows & second column
print("Xtest ", len(X_test))
print(X_test)
print("Ytest ", len(D_test))
print(D_test)
print("Z1test ", len(X1_test))
print(X1_test)
print("Z2test ", len(X2_test))
print(X2_test)

X_train = df_train.iloc[:, :-1].values  # from train DataFrame ..........
D_train = df_train.iloc[:, -1].values
X1_train = df_train.iloc[:, 0].values
X2_train = df_train.iloc[:, 1].values

df_trainY = (
    df_train.loc[df_train["diagnosis"] == 1].iloc[:, :-1].values
)  # np array for positive diagnosis
df_trainN = (
    df_train.loc[df_train["diagnosis"] == 0].iloc[:, :-1].values
)  # np array for negative diagnosis

X_trainY = df_trainY[:, :-1]  # from train DataFrame ......... diagnosis = Yes
D_trainY = df_trainY[:, -1]
X1_trainY = df_trainY[:, 0]
X2_trainY = df_trainY[:, 1]
X_trainN = df_trainN[:, :-1]  # from train DataFrame ......... diagnosis = No
D_trainN = df_trainN[:, -1]
X1_trainN = df_trainN[:, 0]
X2_trainN = df_trainN[:, 1]


print("lenght train: ", len(df_train))
print("lenght train Y: ", len(df_trainY))
print("lenght train N: ", len(df_trainN))

df_train.hist(column="mean radius", bins=nbins, density=1)
df_ax = df_train.hist(
    column="mean radius", by=df_train["diagnosis"], bins=nbins, density=1
)

# train.hist(column='mean radius',bins=nbins,density=1)

fig0, ax0 = plt.subplots(1, 1)
bin_heights, bin_borders, _ = ax0.hist(X1_train, bins=nbins, density=True)
plt.grid(visible="True")
mu0, sigma0 = norm.fit(X1_train)
print("mu0: ", mu0, " sigma0: ", sigma0)
g_x = np.linspace(norm.ppf(0.01, mu0, sigma0), norm.ppf(0.99, mu0, sigma0), 100)
ax0.plot(g_x, norm.pdf(g_x, mu0, sigma0), "r-", lw=5, alpha=0.6, label="norm pdf")
ax0.set_title("mean radius all")
#
fig1, ax1 = plt.subplots(1, 1)
bin_heights, bin_borders, _ = ax1.hist(X1_trainY, bins=nbins, density=True)
plt.grid(visible="True")
mu1, sigma1 = norm.fit(X1_trainY)
print("mu1: ", mu1, " sigma1: ", sigma1)
g_x = np.linspace(norm.ppf(0.01, mu1, sigma1), norm.ppf(0.99, mu1, sigma1), 100)
ax1.plot(g_x, norm.pdf(g_x, mu1, sigma1), "r-", lw=5, alpha=0.6, label="norm pdf")
ax1.set_title("mean radius diagnosis = Yes")
#
fig2, ax2 = plt.subplots(1, 1)
bin_heights, bin_borders, _ = ax2.hist(X1_trainN, bins=nbins, density=True)
plt.grid(visible="True")
mu2, sigma2 = norm.fit(X1_trainN)
print("mu2: ", mu2, " sigma2: ", sigma2)
g_x = np.linspace(norm.ppf(0.01, mu2, sigma2), norm.ppf(0.99, mu2, sigma2), 100)
ax2.plot(g_x, norm.pdf(g_x, mu2, sigma2), "r-", lw=5, alpha=0.6, label="norm pdf")
ax2.set_title("mean radius diagnosis = No")
#


# ---------------------------------------------


data = pd.read_csv(
    r"G:\Il mio Drive\1Master Data Analytics and data science\lezioni_MDA\data analysis\codice\dp-export-414127.csv"
)  # data from local file, only 4 columns

df_train, df_test = train_test_split(data, test_size=0.2, random_state=41)
df_train
df_test

num = len(df_train[df_train.diagnosis == 1])
den = len(df_train.diagnosis)

prior = num / den
prior

import numpy as np

# x = 0.6
media = np.mean(df_train["mean radius"])
devstd = np.std(df_train["mean radius"])

x = 18
# media = 0
# dev = 1
(x - media) / devstd


gauss = (
    1 / (devstd * np.sqrt(2 * np.pi)) * np.exp(-(1 / 2) * ((x - media) / devstd) ** 2)
)
gauss

gauss = 1 / (np.sqrt(2 * np.pi * (devstd ** 2))) * ((x - media) ** 2 / 2 * devstd ** 2)


gauss = (
    1 / (np.sqrt(2 * np.pi * (devstd ** 2))) * (-1 / 2 * (x - media) ** 2 / devstd ** 2)
)

gauss * 1.65

mu = np.mean(df_train["mean radius"])
devstd = np.std(df_train["mean radius"])

x = 18
gaussian = 1 / (devstd * np.sqrt(2 * np.pi) * np.exp(-1 / 2 * (x - mu / devstd)))
gaussian

z = 0
mu = 0
devstd = 1

z = (z - mu) / devstd
norm = 1 / (np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * z ** 2)

1 - norm

c=75
mu=72
devstd=6
z = (c - mu )/ devstd
z

norm = 1 / (np.sqrt(2 * np.pi)) * np.exp((-1 / 2 )* z ** 2)
1- norm