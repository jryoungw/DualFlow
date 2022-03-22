from DualClass import DualNumber, DualTensor
import numpy as np
from time import time


def grad_prop(input : DualTensor, weight : np.ndarray) -> DualTensor:
    # (n, m), (m, k) -> (n, k)
    grad = np.ones_like(weight)
    return DualTensor(np.matmul(input.real, weight), np.matmul(input.dual, weight) + np.matmul(input.real, grad))

def err_prop(input : DualTensor, output : DualTensor) -> np.ndarray:
    return np.outer(input.real, output.dual)

N, M = 500, 800

x_train = np.random.randn(N)
W = np.random.randn(N, M)
y_train = np.matmul(x_train, W)

n_data = len(x_train)
epochs = 100
learning_rate = 1e-4

start = time()

W = np.random.randn(N, M)
x_train = DualTensor(x_train)
x_train.detach()

for i in range(epochs):
    hypothesis = grad_prop(x_train, W)
    hypothesis.dual *= (hypothesis.real - y_train) 
    W -= learning_rate * err_prop(x_train, hypothesis)
    if i % 10 == 0:
        #print(err_prop(hypothesis, W).dual.shape)
        print(np.linalg.norm(hypothesis.real - y_train))

print('W: {}'.format(W))
print(f'Vectorized version : {time() - start} sec took for {epochs} epochs.')

