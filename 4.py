from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_digit(no):
    pred = KNN.predict([x_test[no]])
    img_array = x_test[no].reshape((28,28))
    plt.figure(figsize=(3,3))
    plt.title(f"predicted number ={pred}")
    plt.imshow(img_array)
    
df = pd.read_csv("train.csv")
x = df.values[:,1:]
y = df.values[:,0]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(x_train,y_train)
y_pred = KNN.predict(x_test)
print(y_pred)
a = accuracy_score(y_test,y_pred)
print(f"The accuracy is: {a}")

show_digit(20)
plt.plot(y_test,y_pred)
plt.show()
