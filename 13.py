#13
from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path="IRIS.csv"

data=pd.read_csv(path)
print(data.head())

x=data["Sepal.Length"]
y=data["Species"]
plt.xlabel("sepal length")
plt.ylabel("species")
plt.plot(x,y)
plt.show()

X=data.drop(["Species"],axis=1)
y=data["Species"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=15)

model=KNeighborsClassifier()
model.fit(X,y)

y_pred=model.predict(X_test)
print(y_pred)

a=accuracy_score(y_test,y_pred)
print(f"the accuracy is : {a}")
