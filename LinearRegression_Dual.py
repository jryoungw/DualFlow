from DualClass import DualNumber
import numpy as np
from time import time

x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 12., 15., 18., 21., 24.])

if __name__=="__main__":
    W = DualNumber(0)
    b = DualNumber(0)
    
    # Non vectorized version
    
    n_data = len(x_train)
    epochs = 5000
    learning_rate = 0.01
    
    start = time()
    
    for i in range(epochs):
        cost = 0
        gradient_w = 0
        gradient_b = 0
        for j, t in enumerate(x_train):
            
            hypothesis = t * W + b  
            dual_cost = ((hypothesis - y_train[j]) ** 2)
            gradient_w += dual_cost.dual * t # Forward Differentiation
            gradient_b += dual_cost.dual #     Forward Differentiation

        gradient_w /= n_data
        gradient_b /= n_data
        W -= learning_rate * gradient_w
        b -= learning_rate * gradient_b
    print('W: {}'.format(W.real))
    print('b: {}'.format(b.real))
    print(f'Not vectorized version : {time() - start} sec took for {epochs} epochs.')
    
    # Vectorized version
    
    start = time()
    
    W = DualNumber(0)
    b = DualNumber(0)
    
    cost = DualNumber(0)
    
    gradient_w = 0
    gradient_b = 0
    
    for i in range(epochs):
        hypothesis = (W * x_train + b)
        dual_cost = ((hypothesis - y_train) ** 2)
        gradient_w = (dual_cost.dual * x_train).sum() / n_data
        gradient_b = (dual_cost.dual).sum() / n_data
        
        W -= learning_rate * gradient_w
        b -= learning_rate * gradient_b
    
    print('W: {}'.format(W.real))
    print('b: {}'.format(b.real))
    print(f'Vectorized version : {time() - start} sec took for {epochs} epochs.')