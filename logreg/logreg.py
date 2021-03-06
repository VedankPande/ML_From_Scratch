import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import rand

np.random.seed(0)

#load processed data (cleaned and scaled, outliers not checked)
train_path = 'Desktop/data/BC_train.csv'
df = pd.read_csv(train_path)

# calculate sigmoid function
def sigmoid(X): 
    return 1/(1+ np.e**(-X))

# return cost for logreg
def calc_cost(X,Y,params):
    preds = sigmoid(np.dot(X,params))
    cost1 = Y*(np.log(preds))
    cost2 = (1-Y)*(np.log(1-preds))
    return -((cost1+cost2).mean())

# fit reg line over given data
def fit(X,Y,params,iter,lr):

    for i in range(iter):
        preds = sigmoid(np.dot(X,params))
        for j in range(len(params)):
            params[j] = params[j] - lr*((preds-Y)*X[:,j]).mean()
        
        if i%100 == 0:
            print(f"cost: {calc_cost(X,Y,params)}  params: {params}")
    return params

# select useful features (from correlation heatmap in ipynb)
X = df[['concave points_mean','radius_worst','perimeter_worst','concave points_worst']].iloc[:,:-1]
X_train = np.column_stack((np.ones(X.shape[0]),X))
Y = df.iloc[:,-1]

#hyperparameters
params = np.zeros(X_train.shape[1])
iter = 10000
lr = 0.01

# run
params = fit(X_train,Y,params,iter,lr)

# TODO: make code modular
