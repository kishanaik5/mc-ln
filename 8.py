# ANN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

x = np.array(([2,9],[1,5],[3,6]),dtype= float)
y = np.array(([92],[86],[89]),dtype= float)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x/np.amax(x,axis=0)
y_train = y/100

model = MLPRegressor(hidden_layer_sizes=(32,16),activation='relu',max_iter=100)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

loss = model.loss_

print(f"Loss is {loss}")
print(f"Actual value {y_test}")
print(f"predicted value {y_pred}")

plt.plot(model.loss_curve_,label = "Train")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
