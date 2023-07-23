import numpy as np
from sklearn.linear_model import LinearRegression

hours_studied = np.array([2,3,4,5,6]).reshape(-1,1)
exam_scores = np.array([70,75,85,90,95])
model = LinearRegression()
model.fit(hours_studied,exam_scores)
new = model.predict(np.array([7]).reshape(-1,1))
print(new)
