import pandas as pd
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression


digits=load_digits()
plt.figure(figsize=(16,9))
for index,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 10)


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25, random_state=0)
logicalRegg=LogisticRegression()

logicalRegg.fit(x_train,y_train)
pridict=logicalRegg.predict(x_test)
# print(y_test)
print(pridict[0])
plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
plt.show()
# for i in enumerate(zip(y_test,pridict)):
#     print(i)
