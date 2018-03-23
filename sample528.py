#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

#plt.plot(train_x, train_y, 'o')
#plt.show()

theta0 = np.random.rand()
theta1 = np.random.rand()

def f(x):
    return theta0 + theta1*x

def E(x,y):
    return 0.5 * np.sum(( y - f(x))**2)

mu = train_x.mean()
sigma = train.std()

def standardize(x):
    return (x - mu)/sigma

train_z = standardize(train_x)

theta = np.random.rand(3)

def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x **2 ]).T

X = to_matrix(train_z)

def f(x):
    return np.dot(x, theta)
