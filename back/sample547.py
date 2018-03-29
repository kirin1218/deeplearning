import numpy as np
import matplotlib.pyplot as plt

#   学習データを読み込む
train = np.loadtxt( 'data3.csv', delimiter=',', skiprows=1 )
train_x = train[:,0:2]
train_y = train[:,2]

#   パラメータの初期化
theta = np.random.rand(4)

#   標準化
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)
def standardize(x):
    return (x-mu)/sigma

train_z = standardize(train_x)

#   x0 と x3を追加する
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = 
#   プロット
plt.plot(train_x[train_y == 1, 0],train_x[train_y == 1, 1], 'o' )
plt.plot(train_x[train_y == 0, 0],train_x[train_y == 0, 1], 'x' )
plt.show()
