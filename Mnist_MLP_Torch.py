import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import Tensor
import numpy as np

        
class CustomMM(torch.autograd.Function): ## (b, n, m), (m, k) -> (b, n, k)
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = torch.from_numpy(np.matmul(input.data.numpy(), weight.data.numpy()))

        return output

    @staticmethod
    def backward(ctx, grad_output):  ## (b, n, k) -> (b, n, m), (m, k)
        input, weight = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight.T)
        grad_weight = torch.matmul(input.transpose(1,2), grad_output).sum(0) ## (b, m, n), (b, n, k) -> (b, m, k)
        grad_bias = grad_output.sum(0) ## (b, n, k) -> (n, k)
        return grad_input, grad_weight

class Linear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        mm = CustomMM.apply
        return mm(input, self.weight.T)

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = Linear(128, 10)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        x = F.tanh(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return self.softmax(x).view(-1, 10)
    
model = MnistModel()


batch_size = 50
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=1000)


optimizer = optim.Adam(model.parameters(), lr=0.0001)

#model.load_state_dict(torch.load('test.pth'))
#model.eval()
model.train()
i = 0
for epoch in range(1):
    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()    # calc gradients
        optimizer.step()   # update gradients
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        if i % 100 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data, accuracy))
        i += 1
print('Total accuracy :',accuracy)
#torch.save(model.state_dict(),'test.pth')


