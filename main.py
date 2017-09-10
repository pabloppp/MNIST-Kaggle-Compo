import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deep_nn as nn
import time

np.random.seed(21071969)

ground_truth = pd.read_csv("train.csv").as_matrix().T
X = ground_truth[1:] / 255
Y_raw = ground_truth[:1]
Y = np.eye(10)[Y_raw.reshape(-1)].T  # .map()

train_set_size = 33600

X_train = X[:, :train_set_size]
Y_train = Y[:, :train_set_size]

X_test = X[:, train_set_size:]
Y_test = Y[:, train_set_size:]
Y_raw_test = Y_raw[:, train_set_size:]

m = Y.shape[1]
m_train = Y_test.shape[1]
m_test = Y_test.shape[1]

print("Shape of X", X.shape)
print("Shape of Y", Y.shape)
assert (X.shape == (784, 42000))  # 784 == 28x28
assert (Y.shape == (10, 42000))  # the output is a single label from 0 to 9
# print(Y_raw[:, :5])
# print(Y[:, :5])

input_layer_dims = X.shape[0]
output_layer_dims = Y.shape[0]

# SIMPLE LOGISTIC REGRESSION
parameters = nn.initialize_parameters([input_layer_dims, 249, 171, 63, 25, output_layer_dims])
costs = []
accuracies = []

print("Shape of W1", parameters["W1"].shape)
print("Shape of b1", parameters["b1"].shape)

learning_rate = 0.1
epoch_size = 200
batch_size = train_set_size // 8  # mini batches for the full test data
batch_count = X_train.shape[1] // batch_size

last_epoch_time = time.time()

for i in range(4000):
    # Generate minibatch
    batch_index_start = (i % batch_count) * batch_size
    batch_index_end = batch_index_start + batch_size
    X_batch = X_train[:, batch_index_start:batch_index_end]
    Y_batch = Y_train[:, batch_index_start:batch_index_end]
    # forward propagation
    AL, caches = nn.L_model_forward(X_batch, parameters)
    # calculate costs
    cost = nn.compute_cost(AL, Y_batch)
    costs.append(cost)
    # backward propagation
    grads = nn.L_model_backward(AL, Y_batch, caches)
    # update props
    parameters = nn.update_parameters(parameters, grads, learning_rate)

    # epoch
    if i % epoch_size == 0:
        print("Iteration", i)
        elapsed_time = time.time() - last_epoch_time
        print("Elapsed time since las epoch", "{0:.2f}".format(elapsed_time) + "s",
              "(" + "{0:.2f}".format(elapsed_time / epoch_size) + "s/iter)")
        last_epoch_time = time.time()
        prediction = nn.predict(X_test, parameters)
        print(prediction[:, :30])
        print(Y_raw_test[:, :30])
        accuracy = np.sum(prediction == Y_raw_test) / prediction.shape[1]
        accuracies.append(accuracy)
        print("Cost:", cost)
        print("Accuracy:", accuracy)
        print("--------")

compo_data = pd.read_csv("test.csv").as_matrix().T
X_compo = compo_data / 255
print("X_compo", X_compo.shape)

prediction = nn.predict(X_compo, parameters)

df = pd.DataFrame(prediction.T, columns=['Label'])
df.index += 1
df.to_csv('submission.csv', index=True, index_label='ImageId')

print("DONE!")

plt.figure()
plt.plot(costs)  # plott cost fn
plt.legend(['cost'])

plt.figure()
plt.plot(accuracies)  # plott accuracy fn
plt.legend(['accuracy'])

plt.show()
