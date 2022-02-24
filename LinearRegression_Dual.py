from DualClass import DualNumber
import numpy as np

x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 12., 15., 18., 21., 24.])

if __name__=="__main__":
    W = DualNumber(0)
    b = DualNumber(0)
    n_data = len(x_train)
    epochs = 5000
    learning_rate = 0.01
    
    gradient_w = 0
    gradient_b = 0
    
    for i in range(epochs):
        cost = 0
        for j, t in enumerate(x_train):
            hypothesis = t * W + b  
            cost += ((hypothesis - y_train[j]) ** 2).real
            gradient_w += ((hypothesis - y_train[j]) ** 2).dual * t # Forward Differentiation
            gradient_b += ((hypothesis - y_train[j]) ** 2).dual #     Forward Differentiation

        cost /= n_data
        gradient_w /= n_data
        gradient_b /= n_data
        W -= learning_rate * gradient_w
        b -= learning_rate * gradient_b
    print('W: {}'.format(W))
    print('b: {}'.format(b))