import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

iris_df = pd.read_csv("IRIS.csv")
print(iris_df.head())
print("\n")
print(iris_df.count())
print("\n")
print(iris_df.isnull().sum())
print('\n')
iris = iris_df.drop(['ID'],axis=1)
iris.plot()
plt.show()
print("\n")
cov_mat = iris.cov()
print(cov_mat)
print("\n")
corr_mat = iris.corr()
print(corr_mat)
print("\n")
X = iris.drop(["Species"],axis=1)
y = iris["Species"]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 10)
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)
a = accuracy_score(y_test,y_pred)
print(f"the accuracy is:{a}")
