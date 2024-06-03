import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
bosten=load_boston()

df=pd.DataFrame(bosten.data,columns=bosten.feature_names)

df['MEDV']=bosten.target
y=df['MEDV']
x=pd.DataFrame(np.c_[df['LSTAT'],df['RM']],columns=['LSTAT','RM'])
print(df.isnull().sum())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


linearregg=LinearRegression()
linearregg.fit(x_train,y_train)

pridict=linearregg.predict(x_test)
print(pridict.shape)
plt.scatter(x_test['LSTAT'],pridict)
plt.show()