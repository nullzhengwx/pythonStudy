import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class AdalineGD(object):
    """ ADAptive Linear NEuron (Adaline)
        classifier

    Parameters
    ------------
    eta : float
        Learning rate ( between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X , y):
        """ Fit traing data.

        Parameters:
        --------------
        :param X:  {array-like}, shape = [n_samples, n_features]
                    Training vectors,
                    where n_samples is the number of samples and
                    n_features is the number of features.
        :param y:  array-like, shape = [n_samples]
                    Target values.
        :return:
        -----------
        self : Object
        """

        self.w_ =  np.zeros(1 + X.shape[1])     # [w0, w1, w2.....wn]
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """ Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ Compute linear activation """
        return self.net_input(X)

    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.activation(X) >= 0.0, 1, -1)

""" testing and training """
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris/data', header=None)
df.tail()


y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0:100, [0, 2]].values

fig, ax = plt.subplot(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)

ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)

ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='0')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

""" Standardization : standerd normal distribution using NumPy"""
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()