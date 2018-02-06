import numpy as np

bias = 4
X = np.array([[2., 3.],
              [4., 5.],
              [6., 7.]])
w = np.array([bias, 2., 3.])

ones = np.ones((X.shape[0], 1))
X_with1 = np.hstack((ones, X))
print(X_with1)
print(np.dot(X_with1, w))

Y = np.array([[3., 5.],
              [7., 20.]])

X_with_Y = np.vstack((X, Y))
print(X_with_Y)