import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#set default options
sns.set()   
np.seterr('raise')

# set up data (not processed)
path = "Desktop/data/housingdata.csv"
df = pd.read_csv(path)
X = df[['RM','DIS']]
X = np.column_stack((np.ones(X.shape[0]),X))
Y = df['MEDV']

#create zero param array
params = np.zeros(X.shape[1])

iter = 10000
lr = 0.0047
cost_log = []

#fit model
for i in range(iter):
    preds = np.dot(X,params)

    try:
        params[0] = params[0] - lr*(preds-Y).mean()
        params[1] = params[1] - lr*((preds-Y)*X[:,1]).mean()
        params[2] = params[2] - lr*((preds-Y)*X[:,2]).mean()
        cost = ((preds-Y)**2).mean()/2
    except Exception as e:
        print(e)
        break
    if i%100 == 0:
        print(f"epoch:{i+1}  params : {params}  cost: {cost}")
        cost_log.append(cost)

#plot cost
plt.scatter(np.arange(len(cost_log)),cost_log)
plt.show()

