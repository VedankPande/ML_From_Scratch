import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

#dataframes
train_path = 'C:/Users/anike/Desktop/data/iris/IRIS_train.csv'
train_df = pd.read_csv(train_path)
test_path = 'C:/Users/anike/Desktop/data/iris/IRIS_test.csv'
test_df = pd.read_csv(test_path)

# Euclidean distance
def calc_distance(inst1,inst2):
    inst1 = np.array(inst1)
    inst2 = np.array(inst2)
    return np.linalg.norm(inst1-inst2)

#calculate distance for all points from test points
#TODO: Try to optimize this
distance_log = []
for i in tqdm(range(len(train_df))):
    distance_log.append([i,calc_distance(test_df.iloc[0,:-3],train_df.iloc[i,:-3])])
k = int(input("enter k value"))

#sort data based on distance and get closest k neighbours
subset = sorted(distance_log,key= lambda item : item[1])[:k]

#prints out classes for closest data points
labels = [val[0] for val in subset]
for idx in test:
    print(np.array(train_df.iloc[idx,-3:]))