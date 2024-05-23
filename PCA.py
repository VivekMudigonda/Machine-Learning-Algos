import numpy as np
from keras.datasets import mnist
from PIL import Image as im
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X = X_train[y_train == 2][:100].reshape(-1, 28 * 28)
test_image = X_test[y_test == 2][0].reshape(28 * 28, 1)

X = X.T
d, n = X.shape

def PrintImage(X):
    y = np.arange(0, 784, 1, np.uint8)
    y = np.resize(X, (28, 28))
    data = im.fromarray(y)

    plt.imshow(np.squeeze(data), cmap=plt.cm.binary)
    plt.show()

def Caltopeigen(eighval):
    lambdaTotal = np.sum(eighval)
    k = 0
    for i in range(len(eighval)):
        k += eighval[len(eighval) - i - 1]
        if (k / lambdaTotal) * 100 >= 95:
            break
    print(i + 1)
    print((k / lambdaTotal) * 100)

mean = np.mean(X, axis=1)
mean = np.expand_dims(mean, axis=1)
X_centered = X - mean

# Calculating the covariance matrix
C = (X_centered @ X_centered.T) / n

# Finding the eigen vectors of the covariance matrix
eighval, eighvec = np.linalg.eigh(C)
w_1, w_2 = eighvec[:, -1], eighvec[:, -2]
w_1 = w_1.reshape(w_1.shape[0], 1)
w_2 = w_2.reshape(w_2.shape[0], 1)
Caltopeigen(eighval)

lambdaTotal = np.sum(eighval)

def VarCalc(Y, Varx, x):
    k = eighval[-1]
    for i in range(len(eighval) - 1):
        k += eighval[len(eighval) - i - 2]
        w = eighvec[:, len(eighval) - i - 2]
        w = w.reshape(w.shape[0], 1)
        Varx += w @ (w.T @ Y)
        if (k / lambdaTotal) * 100 >= x:
            break

def Plot(fig, Y, i, s, rows, columns):
    ax = fig.add_subplot(rows, columns, i)
    y = np.resize(Y, (28, 28))
    ax.imshow(y, cmap=plt.cm.binary)
    ax.axis('off')
    ax.set_title(s)

def PrintTest(Y):
    print(Y.shape)
    Var20 = np.zeros(Y.shape)
    Var50 = np.zeros(Y.shape)
    Var80 = np.zeros(Y.shape)
    Var95 = np.zeros(Y.shape)
    Var100 = np.zeros(Y.shape)

    VarCalc(Y, Var20, 20)
    VarCalc(Y, Var50, 50)
    VarCalc(Y, Var80, 80)
    VarCalc(Y, Var95, 95)

    k = eighval[-1]
    for i in range(len(eighval) - 2):
        w = eighvec[:, len(eighval) - i - 2]
        w = w.reshape(w.shape[0], 1)
        Var100 += w @ (w.T @ Y)

    rows = 1
    columns = 5

    fig = plt.figure(figsize=(15, 5))

    Plot(fig, Var20, 1, "Var20", rows, columns)
    Plot(fig, Var50, 2, "Var50", rows, columns)
    Plot(fig, Var80, 3, "Var80", rows, columns)
    Plot(fig, Var95, 4, "Var95", rows, columns)
    Plot(fig, Var100, 5, "Var100", rows, columns)

    plt.show()

# You can use any other test image with same shape
PrintTest(test_image)
