import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

img_size = 100

class Net(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.conv1 = nn.Conv2d(1,32,kernel_size=5)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5)
        self.conv3 = nn.Conv2d(64,128,kernel_size=5)

        #figure out input size
        self.fc1 = nn.Linear(128*9*9,512)
        self.fc2 = nn.Linear(512,1024)
        self.fc3 = nn.Linear(1024,2)

    def forward(self, x):
        
       x = F.avg_pool2d(F.relu(self.conv1(x)), (2,2))
       #print(f'shape after conv1: {x.shape}')
       x = F.avg_pool2d(F.relu(self.conv2(x)), (2,2))
       #print(f'shape after conv1: {x.shape}')
       x = F.avg_pool2d(F.relu(self.conv3(x)), (2,2))
       print(f'shape after conv1: {x.shape}')
       #sys.exit('get shape')
       x = x.view(-1,128*9*9)

       x = F.relu(self.fc1(x))
       x = F.celu(self.fc2(x))
       x = self.fc3(x)

       x = F.softmax(x, dim=1)

       return(x)

net = Net()

test_img = torch.randn(img_size,img_size).view(-1,1,img_size,img_size)
output = net(test_img)