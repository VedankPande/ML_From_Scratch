import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()   
np.seterr('raise')
path = "Desktop/data/housingdata.csv"
df = pd.read_csv(path)
X = df[['RM','DIS']]
X = np.column_stack((np.ones(X.shape[0]),X))
Y = df['MEDV']

params = np.zeros(X.shape[1])

iter = 10000
lr = 0.0047
cost_log = []
for i in range(iter):
    #print(f"epoch: {i+1}")
    preds = np.dot(X,params)
    
    #for j in range(len(params)):
    #    params[j] = params[j] - lr*((preds-Y)*X[:,j]).mean()
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

plt.scatter(np.arange(len(cost_log)),cost_log)
plt.show()

