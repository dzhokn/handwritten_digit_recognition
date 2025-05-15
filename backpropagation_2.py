import numpy as np
import pandas as pd

# Load the MNIST dataset
data = pd.read_csv("data/mnist_train.csv")

# Data preprocessing
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Normalize the values of the pixels to be between 0 and 1
Y_train = data.T[0]
X_train = data.T[1:]
X_train = X_train / 255.

# Initialize parameters
def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

# Define activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Forward propagation
def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# Backpropagation
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def deriv_ReLU(Z):
    return Z > 0

def back_prop(z1, a1, z2, a2, w2, Y, X):
    OneHot_Y = one_hot(Y)
    dZ2 = a2 - OneHot_Y
    dW2 = (dZ2 @ a1.T) / m
    db2 = np.sum(dZ2) / m

    dZ1 = (w2.T @ dZ2) * deriv_ReLU(z1) # Multiply the derivative of the ReLU function by the dot product of the weights and the error
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1) / m

    return dW1, db1, dW2, db2

# Parameter updates
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

# Gradient descent
def get_predictions(a2):
    return np.argmax(a2,0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2=init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = back_prop(z1, a1, z2, a2, w2, Y, X)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print(f"Iteration #{i} accuracy: {get_accuracy(get_predictions(a2),Y)}")
    return w1, b1, w2, b2

# Training the model
w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 1000, 0.15)