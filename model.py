# Import needed module
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Defining needed function(s)
def multiple_linear_regression(x, w):
    return x @ w


def loss_function(y, y_hat):
    return np.linalg.norm((y_hat - y))**2 / len(y)


def r_squared(y, y_hat):
    return 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)


# Reading data
data = pd.read_csv(r'BostonHousing.csv')
data = np.array(data)

# Seperating data and label
data_sample = data[:, :-1]
data_label = data[:, -1]

# Data normalization
x_train, x_test, y_train, y_test = train_test_split(data_sample, data_label, test_size= 0.2, random_state= 12)

# Data normalization
standard = StandardScaler()
x_train_scale = standard.fit_transform(x_train)
x_test_scale = standard.transform(x_test)

# Training model
# Augmenting train data
n, m = x_train_scale.shape
x_train_aug = np.hstack((np.ones((n, 1)), x_train_scale))

# Acquire best weights
w = np.linalg.inv(x_train_aug.T @ x_train_aug) @ x_train_aug.T @ y_train

# Testing model
# Augmenting test data
n, m = x_test_scale.shape
x_test_aug = np.hstack((np.ones((n, 1)), x_test_scale))

# Acquire y_hat
y_hat_test = multiple_linear_regression(x_test_aug, w)

# Loss
r2 = r_squared(y_test, y_hat_test)
print(f"Model accuracy: {r2*100:.2f}%")