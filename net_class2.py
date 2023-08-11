import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class Net(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.conv1 = nn.Conv1d(1,32,kernel_size=9)
        self.conv2 = nn.Conv1d(32,64,kernel_size=9)

        #figure out input size
        self.fc1 = nn.Linear(9216,1024)
        self.fc2 = nn.Linear(1024,2048)
        self.fc3 = nn.Linear(2048,2)

    def forward(self, x):
        
       x = F.max_pool1d(F.relu(self.conv1(x)), (2))
       x = F.max_pool1d(F.relu(self.conv2(x)), (2))
       #print(f'shape after conv1: {x.shape}')
       x = x.view(-1,9216)

       x = F.relu(self.fc1(x))
       x = F.celu(self.fc2(x))
       x = self.fc3(x)

       x = F.softmax(x, dim=1)

       return(x)

net = Net()

test_img = torch.randn(602).view(-1,1,602)
output = net(test_img)