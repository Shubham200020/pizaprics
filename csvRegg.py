import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df=pd.read_csv('pizza_v1.csv')


df['size']=df['size'].replace(to_replace='jumbo',value=0)
df['size']=df['size'].replace(to_replace='reguler',value=1)
df['size']=df['size'].replace(to_replace='small',value=2)
df['size']=df['size'].replace(to_replace='medium',value=3)
df['size']=df['size'].replace(to_replace='large',value=4)
df['size']=df['size'].replace(to_replace='XL',value=5)


df['company']=df['company'].replace(to_replace='A',value=0)
df['company']=df['company'].replace(to_replace='B',value=1)
df['company']=df['company'].replace(to_replace='C',value=2)
df['company']=df['company'].replace(to_replace='D',value=3)
df['company']=df['company'].replace(to_replace='E',value=4)


df['extra_cheese']=df['extra_cheese'].replace(to_replace='no',value=0)
df['extra_cheese']=df['extra_cheese'].replace(to_replace='yes',value=1)

df['extra_sauce']=df['extra_sauce'].replace(to_replace='no',value=0)
df['extra_sauce']=df['extra_sauce'].replace(to_replace='yes',value=1)


list=df['price_rupiah']

list=[]
y=[]
for i in df['price_rupiah']:
    list.append(i.split('Rp')[1])
    

for i in list:
    y.append(int((i.split(',')[0])+(i.split(',')[1])))

x=np.c_[df['size'],df['diameter'],df['company'],df['extra_cheese'],df['extra_sauce']]
df['price_rupiah']=y
y=df['price_rupiah']

print(df.company.unique())
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
linearregg=LogisticRegression()

linearregg.fit(x_train,y_train)
print(x_test)
print("Actual")
print(y_test)
print("Pridicted")
pridict=linearregg.predict(x_test)
print(pridict)
print(x_test.shape)
print(y_test.shape)

print(np.sqrt(metrics.mean_squared_error(y_test, pridict)))

