import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(21071969)

ground_truth = pd.read_csv("train.csv").as_matrix().T
X = ground_truth[1:] / 255
Y_raw = ground_truth[:1]
Y = np.eye(10)[Y_raw.reshape(-1)].T  # .map()

X_train = X[:, :33600]
Y_train = Y[:, :33600]

X_test = X[:, 33600:]
Y_test = Y[:, 33600:]
Y_raw_test = Y_raw[:, 33600:]

m = Y.shape[1]
m_train = Y_test.shape[1]
m_test = Y_test.shape[1]

print("Shape of X", X.shape)
print("Shape of Y", Y.shape)
assert (X.shape == (784, 42000))  # 784 == 28x28
assert (Y.shape == (10, 42000))  # the output is a single label from 0 to 9
# print(Y_raw[:, :5])
# print(Y[:, :5])


# SIMPLE LOGISTIC REGRESSION
W1 = np.random.randn(Y.shape[0], X.shape[0]) * 0.001
b1 = np.zeros((Y.shape[0], 1))

print("Shape of W1", W1.shape)
print("Shape of b1", b1.shape)
assert (W1.shape == (10, 784))
assert (b1.shape == (10, 1))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagation(x, w, b):
    z = np.dot(w, x) + b
    a = sigmoid(z)
    return a


def calculate_loss(a, y, m):
    first = np.multiply(y, np.log(a))
    second = np.multiply(1 - y, np.log(1 - a))
    l = - (1 / m) * np.sum(first + second, axis=1, keepdims=True)
    return l


def calculate_gradients(x, a, y, m):
    dw = (1 / m) * np.dot(x, (a - y).T)
    db = (1 / m) * np.sum(a - y, axis=1, keepdims=True)
    return dw.T, db


def back_propagation(w, b, dw, db, learning_rate=0.01):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b


def predict(x, w, b):
    a = forward_propagation(x, w, b)
    prediction = np.argmax(a, axis=0).reshape(1, a.shape[1])
    return prediction


epoch_size = 100
batch_size = 10500  # 4 mini batches for the full test data
batch_count = X_train.shape[1] // batch_size

costs = []
accuracies = []

for i in range(8000):
    batch_index_start = (i % batch_count) * batch_size
    batch_index_end = batch_index_start + batch_size
    X_batch = X_train[:, batch_index_start:batch_index_end]
    Y_batch = Y_train[:, batch_index_start:batch_index_end]
    A1 = forward_propagation(X_batch, W1, b1)
    # print("Shape of A1", A1.shape)
    L = calculate_loss(A1, Y_batch, m_train)
    cost = np.squeeze(np.sum(L) / 10)
    costs.append(cost)
    dW1, db1 = calculate_gradients(X_batch, A1, Y_batch, m_train)
    # print("Gradients - dw:", dW1.shape, "db:", db1.shape)
    W1, b1 = back_propagation(W1, b1, dW1, db1)
    if i % epoch_size == 0:
        print("IT:", i)
        print("Cost", cost)
        P = predict(X_test, W1, b1)

        print(P[:, :20])
        print(Y_raw_test[:, :20])
        accuracy = np.sum(P == Y_raw_test) / P.shape[1]
        accuracies.append(accuracy)
        print("Accuracy:", accuracy)

        # print(A1[:, [0]])

plt.figure()
plt.plot(costs)  # plott cost fn
plt.legend(['cost'])

plt.figure()
plt.plot(accuracies)  # plott accuracy fn
plt.legend(['accuracy'])

plt.show()
