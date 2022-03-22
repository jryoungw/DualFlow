import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from DualClass import DualTensor

mnist = loadmat("mnist-original.mat")

X = mnist["data"].T / 256
y = mnist["label"][0]

y = np.eye(10)[y.astype(int)]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

batch_size = 50
input_layer = 784
hidden_1 = 128
output_layer = 10

lr = 1e-3

params = {
    'W1' : np.random.randn(input_layer, hidden_1) * np.sqrt(1. / hidden_1),
    'W2' : np.random.randn(hidden_1, output_layer) * np.sqrt(1. / output_layer),
}

def grad_prop(input : DualTensor, weight : np.ndarray) -> DualTensor:
    grad = np.ones_like(weight)
    return DualTensor(np.matmul(input.real, weight), np.matmul(input.real, grad) + np.matmul(input.dual, weight))


def feed_foward(X):
    params['A0'] = DualTensor(X.reshape(batch_size, 1, -1))
    params['A0'].detach()

    params['A1'] = grad_prop(params['A0'], params['W1'])
    params['A1'] = params['A1'].tanh()

    params['A2'] = grad_prop(params['A1'], params['W2'])
    params['A2'] = params['A2'].tanh()
    return params['A2'].real
'''
input.errorin @ weight = output.errorout
    def backward(ctx, error_out : DualTensor) -> tuple(DualTensor, torch.Tensor):  
        input_real, weight = ctx.saved_tensors
        grad_weight = torch.matmul(input_real.transpose(1,2), error_out.dual)
        error_in = error_out.matmul(DualTensor(weight).inv().T())
        return error_in, grad_weight
'''

def backward_pass(y):

    change = {}
    params['A2'].dual = params['A2'].real - y.reshape(batch_size, 1, -1)

    change['W2'] = np.matmul(params['A1'].real.transpose(0,2,1), params['A2'].dual)
    #params['A1'] = params['A2'].matmul(DualTensor(params['W2']).inv().T())
    params['A1'].dual = np.matmul(params['A2'].dual, params['W2'].T)

    change['W1'] = np.matmul(params['A0'].real.transpose(0,2,1), params['A1'].dual)


    return change


def compute_accuracy():
    prediction = []
    for idx in range(len(x_test)//batch_size):
        xs = x_test[batch_size * idx:batch_size * idx + batch_size]
        ys = y_test[batch_size * idx:batch_size * idx + batch_size]
        outputs = feed_foward(xs)
        for output, y in zip(outputs, ys):
            prediction.append(np.argmax(output) == np.argmax(y))
    return np.mean(prediction) * 100

toperr = 0
for epoch in range(1):
    i = 0
    start = time.time()
    for idx in range(len(x_train)//batch_size):
        x = x_train[batch_size * idx:batch_size * idx + batch_size]
        y = y_train[batch_size * idx:batch_size * idx + batch_size]
        feed_foward(x)
        change = backward_pass(y)

        params['W1'] -= lr * change['W1'].sum(0)
        params['W2'] -= lr * change['W2'].sum(0)
        if i % 100 == 0:
            print('Train Step: {}\tAccuracy: {:.3f}'.format(i, compute_accuracy()))
        i += 1

    print('Acc :', compute_accuracy(), 'Time :',time.time()-start)
