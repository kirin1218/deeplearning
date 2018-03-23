# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def E(x,y):
    return 0.5 * np.sum(( y - f(x))**2)

def standardize(x):
    return (x - mu)/sigma

#   学習データの行列を作る
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x**2]).T

#   予測関数
def f(x):
    return np.dot(x, theta)

train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

mu = train_x.mean()
sigma = train.std()

train_z = standardize(train_x)

ETA = 1e-3

#   パラメータを初期化
theta = np.random.rand(3)


X = to_matrix(train_z)


#   誤差の差分
diff = 1

#   学習を繰り返す
error = E(X, train_y)
while diff > 1e-2:
    #   パラメータを更新
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    #   前回の誤差と差分を計算
    current_error = E(X,train_y)
    diff = error - current_error
    error = current_error
#    print(diff)

x = np.linspace(-3,3,100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()
