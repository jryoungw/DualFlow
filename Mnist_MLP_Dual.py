import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import Tensor
from DualClass import DualTensor

class DualMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_real, input_dual, weight): 
        output_real = F.tanh(torch.matmul(input_real, weight))
        output_dual = (torch.matmul(input_real, torch.ones_like(weight)) + torch.matmul(input_dual, weight)) * (1 - output_real ** 2)
        ctx.save_for_backward(input_real, weight) 
        return output_real, output_dual

    @staticmethod
    def backward(ctx, _, error_out_dual):  
        input_real, weight = ctx.saved_tensors
        grad_weight = torch.matmul(input_real.transpose(1,2), error_out_dual)
        error_out_dual = torch.matmul(error_out_dual, weight.T)
        return None, error_out_dual, grad_weight


class Linear(nn.Linear):
    def forward(self, input_real: Tensor, input_dual: Tensor):
        mm = DualMM.apply
        return mm(input_real, input_dual, self.weight.T)

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        x, grad = self.fc1(x, torch.zeros_like(x))
        x = F.dropout(x, training=self.training)
        x, grad = self.fc2(x, grad)
        return x.view(-1, 10), grad
    
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
        optimizer.zero_grad()
        output, grad = model(data)
        #loss = F.nll_loss(output, target)
        #loss.backward()    # calc gradients
        grad.backward(gradient=(output - torch.eye(10)[target]).reshape(-1, 1, 10))
        optimizer.step()   # update gradients
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        if i % 100 == 0:
            print('Train Step: {}\tAccuracy: {:.3f}'.format(i, accuracy))
        i += 1
print('Total accuracy :',accuracy)
#torch.save(model.state_dict(),'test.pth')


