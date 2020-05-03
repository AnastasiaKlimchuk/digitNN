import pandas as pd
import numpy as np


N_CLASSES = 10


def one_hot(Y, n_classes):
    print(Y.shape)
    one_hot_y = np.zeros((Y.shape[1], n_classes))
    rows = np.arange(Y.shape[1])
    one_hot_y[rows, Y] = 1
    return one_hot_y.T


def read_data(url_train, url_test):
    # Read data
    train = pd.read_csv(url_train)
    labels = train.iloc[:, 0].values.astype('int32')
    X_train = (train.iloc[:, 1:].values).astype('float32')
    X_test = (pd.read_csv(url_test).values).astype('float32')

    return labels, X_train, X_test


def shuffle_data(features, labels, random_seed=42):
    assert len(features) == len(labels)

    if random_seed:
        np.random.seed(random_seed)
    idx = np.random.permutation(len(features))
    return [a[idx] for a in [features, labels]]


def split_test(X_train, y_train):
    X, y = shuffle_data(X_train, y_train, random_seed=42)
    train_set_x, train_set_y = X[:35000], y[:35000]
    test_set_x, test_set_y = X[35000:], y[35000:]

    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
    return train_set_x, train_set_y, test_set_x, test_set_y

#Activation functions
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z


def tanh(Z):
    A = np.tanh(Z)
    return A, Z


def relu(Z):
    A = np.maximum(0, Z)
    return A, Z


def leaky_relu(Z):
    A = np.maximum(0.1 * Z, Z)
    return A, Z

#Gradient of activation function


def sigmoid_gradient(dA, Z):
    A, Z = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ


def tanh_gradient(dA, Z):
    A, Z = tanh(Z)
    dZ = dA * (1 - np.square(A))

    return dZ


def relu_gradient(dA, Z):
    A, Z = relu(Z)
    dZ = np.multiply(dA, np.int64(A > 0))

    return dZ


def accuracy(pred, labels):
    return (np.sum(pred == labels, axis=1) / float(labels.shape[1]))[0]




