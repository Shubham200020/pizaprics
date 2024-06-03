from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
btnData=load_boston()
print(btnData.feature_names)
df=pd.DataFrame(btnData.data,columns=btnData.feature_names)
print(df)
print(df.head())
df['MEDV']=btnData.target
x=pd.DataFrame(np.c_[df['LSTAT'],df['RM']],columns=['LSTAT','RM'])
y=pd.DataFrame(np.c_[df['MEDV']])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print("X Train:",x_train.shape)
print("Y Train:",y_train.shape)
print("x Test",x_test.shape)
print("y Test",y_test.shape)
linregg=LinearRegression()
linregg.fit(x_train,y_train)
pridict_y=linregg.predict(x_test)
print('Pridict')
print(pridict_y)
print("Y test")
print(y_test)
rms=(np.sqrt(mean_squared_error(y_test,pridict_y)))
print(rms)
print(df['RM'])