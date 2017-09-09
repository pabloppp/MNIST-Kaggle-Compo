import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deep_nn as nn

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

input_layer_dims = X.shape[0]
output_layer_dims = Y.shape[0]

# SIMPLE LOGISTIC REGRESSION
parameters = nn.initialize_parameters([input_layer_dims, 250, 64, output_layer_dims])
costs = []
accuracies = []

print("Shape of W1", parameters["W1"].shape)
print("Shape of b1", parameters["b1"].shape)

learning_rate = 0.1
epoch_size = 200
batch_size = 10500  # 4 mini batches for the full test data
batch_count = X_train.shape[1] // batch_size

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
        prediction = nn.predict(X_test, parameters)
        print(prediction[:, :20])
        print(Y_raw_test[:, :20])
        accuracy = np.sum(prediction == Y_raw_test) / prediction.shape[1]
        accuracies.append(accuracy)
        print("Cost:", cost)
        print("Accuracy:", accuracy)
        print("--------")

plt.figure()
plt.plot(costs)  # plott cost fn
plt.legend(['cost'])

plt.figure()
plt.plot(accuracies)  # plott accuracy fn
plt.legend(['accuracy'])

plt.show()


# epoch_size = 100
# batch_size = 10500  # 4 mini batches for the full test data
# batch_count = X_train.shape[1] // batch_size
#
# costs = []
# accuracies = []
#
# for i in range(20000):
#     batch_index_start = (i % batch_count) * batch_size
#     batch_index_end = batch_index_start + batch_size
#     X_batch = X_train[:, batch_index_start:batch_index_end]
#     Y_batch = Y_train[:, batch_index_start:batch_index_end]
#     A1 = forward_propagation(X_batch, W1, b1)
#     # print("Shape of A1", A1.shape)
#     L = calculate_loss(A1, Y_batch, m_train)
#     cost = np.squeeze(np.sum(L) / 10)
#     costs.append(cost)
#     dW1, db1 = calculate_gradients(X_batch, A1, Y_batch, m_train)
#     # print("Gradients - dw:", dW1.shape, "db:", db1.shape)
#     W1, b1 = back_propagation(W1, b1, dW1, db1)
#     if i % epoch_size == 0:
#         print("IT:", i)
#         print("Cost", cost)
#         P = predict(X_test, W1, b1)
#
#         print(P[:, :20])
#         print(Y_raw_test[:, :20])
#         accuracy = np.sum(P == Y_raw_test) / P.shape[1]
#         accuracies.append(accuracy)
#         print("Accuracy:", accuracy)
#
#         # print(A1[:, [0]])
#
# compo_data = pd.read_csv("test.csv").as_matrix().T
# X_compo = compo_data / 255
# print("X_compo", X_compo.shape)
#
# P = predict(X_compo, W1, b1)
#
# df = pd.DataFrame(P.T, columns=['Label'])
# df.index += 1
# df.to_csv('submission.csv', index=True, index_label='ImageId')
