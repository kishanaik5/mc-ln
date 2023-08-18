#14
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate sample data with a known curve (linear or non-linear)
# For this example, we'll use a non-linear curve y = sin(x) + noise
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(80)

# Step 2: Set the value for the smoothing parameter (t)
t = 0.1

# Step 3: Set the bias/point of interest (x0)
x0 = 2.5

# Step 4: Determine the weight matrix using Gaussian Kernel
weights = np.exp(-0.5 * ((X - x0) / t) ** 2)
print(weights)

# Step 5: Determine the value of model term parameter B using Locally Weighted Regression
X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
W = np.diag(weights.ravel())
B = np.linalg.inv(X_with_bias.T @ W @ X_with_bias) @ X_with_bias.T @ W @ y

# Step 6: Prediction
x0_with_bias = np.array([1, x0])
prediction = x0_with_bias @ B

print("Prediction at x0:", prediction)
