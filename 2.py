import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("housing.csv")
print(df.head())
print("\n")
print("The count is:\n")
count = df.count()
print(count)
print("\nNull values in each attributes:\n")
print(df.isnull().sum())
df.plot()
plt.show()
covmat = df.cov(numeric_only = True)
print("\n The Covariance matrix is:\n")
print(covmat)
corrmat = df.corr(numeric_only = True)
print("\n The Correlation matrix is:\n")
print(corrmat)
x = df.drop(["median_house_value"],axis = 1)
y = df["median_house_value"]
x_update = pd.get_dummies(x,columns=['ocean_proximity'])
x_update.fillna(df['total_bedrooms'].mean(),inplace = True)

x_train,x_test,y_train,y_test = train_test_split(x_update,y,test_size=0.2,random_state = 42)
model = SGDRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)

p = mean_squared_error(y_test,y_pred)
a = 1 - (p/np.var(y_test))
print(f"the accuracy of the model is: {a}")

plt.plot(y_test[1:10],y_pred[1:10])
plt.xlabel("actual value")
plt.ylabel("predicted value")
plt.show()
