import numpy as np
import matplotlib.pyplot as plt

#	学習データを読み込む
train = np.loadtxt( 'data3.csv', delimiter=',', skiprows=1 )
train_x = train[:,0:2]
train_y = train[:,2]

#	パラメータを初期化
theta = np.random.rand(4)

#	標準化
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)
def standardize(x):
	return (x-mu)/sigma

train_z = standardize(train_x)

#	x0を加える
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = x[:,0,np.newaxis] ** 2
    return np.hstack([x0, x, x3])

x = to_matrix(train_z)


#   シグモイド関数
def f(x):
    return 1/(1+np.exp(-np.dot(x,theta)))

#   学習率
ETA = 1e-3

#   繰り返し回数
epoch = 5000

#   精度の履歴
accuracies = []

def classify(x):
    return (f(x) >= 0.5).astype(np.int)

for _ in range(epoch):
    theta = theta - ETA*np.dot(f(x)-train_y, x)
    #   現在の制度を計算
    result = classify(x) == train_y
    accuracy = len(result[result == True]) / len(result)
    accuracies.append(accuracy)

#   精度をプロット
x = np.arange(len(accuracies))

plt.plot( x, accuracies)
plt.show()
