theta0 = np.random.rand()
theta1 = np.random.rand()

def f(x):
    return theta0 * theta1 * x

def E(x,y):
    return 0.5 * np.sum((y-f(x))**2)
