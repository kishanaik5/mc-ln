# 7ID3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

path = "testtennis.csv"
data = pd.read_csv(path)

X = data.drop('playtennis', axis=1)
y = data['playtennis']

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)

new_sample = X_test.iloc[[0]]
predicted_class = decision_tree.predict(new_sample)

print("Predicted class for the new sample:", predicted_class[0])

plt.figure(figsize=(15, 10))
plot_tree(decision_tree, feature_names=X_encoded.columns, class_names=['No', 'Yes'])
plt.show()
