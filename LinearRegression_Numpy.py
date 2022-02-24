from DualClass import DualNumber
import numpy as np

x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 12., 15., 18., 21., 24.])

if __name__=="__main__":
    W = 0.0
    b = 0.0

    n_data = len(x_train)

    epochs = 5000
    learning_rate = 0.01

    for i in range(epochs):
        hypothesis = x_train * W + b
        cost = np.sum((hypothesis - y_train) ** 2) / n_data
        gradient_w = np.sum((W * x_train - y_train + b) * 2 * x_train) / n_data # BackPropagation
        gradient_b = np.sum((W * x_train - y_train + b) * 2) / n_data #           BackPropagation

        W -= learning_rate * gradient_w
        b -= learning_rate * gradient_b

    print('W: {}'.format(W))
    print('b: {}'.format(b))