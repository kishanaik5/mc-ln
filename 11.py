#11
from sklearn.ensemble import RandomForestRegressor
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

path="housing.csv"

#import datset

data=pd.read_csv(path)
print(data.head()) #displaying 5 rows of data


count=data.count()
print(count)

#to print number of null values
print(data.isnull().sum())

#graph representation
data.plot()
plt.show()

#cov matrix and corr matrix
cov_mat=data.cov(numeric_only=True)
corr_mat=data.corr(numeric_only=True)
print(cov_mat)
print(corr_mat)

#train and test model

X=data.drop(["median_house_value"],axis=1)
y=data["median_house_value"]
X_encoded = pd.get_dummies(X, columns=['ocean_proximity'])
X_encoded.fillna(data["total_bedrooms"].mean(), inplace=True)

X_train,X_test,y_train,y_test=train_test_split(X_encoded,y,test_size=0.09,random_state=42)

model=RandomForestRegressor()
model.fit(X_train,y_train)

#predicting values

y_pred=model.predict(X_test)
print(y_pred)


#accuracy and its graph
mse = mean_squared_error(y_test, y_pred)
a = 1 - (mse / np.var(y_test))

print(f"the accuracy of the model is : {a}")

plt.plot(y_test[1:10],y_pred[1:10])
plt.xlabel("actual value")
plt.ylabel("predicted value")
plt.show()
