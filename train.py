import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from net_class import Net

img_size = 100

training_data = np.load('melanoma_training_data.npy', allow_pickle=True)

train_X = torch.tensor( [item[0] for item in training_data], dtype=torch.float )
train_y = torch.tensor( [item[1] for item in training_data], dtype=torch.float )

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_function = nn.MSELoss()

batch_size = 12
epochs = 2

for epoch in range(epochs):
    for i in range(0,len(train_X), batch_size):
        print(f'EPOCH: {epoch +1}, fraction complete: {i/len(train_X)}')

        batch_X = train_X[i: i+batch_size].view(-1, 1, img_size, img_size)
        batch_y = train_y[i: i+batch_size]

        optimizer.zero_grad()

        outputs = net(batch_X)

        loss = loss_function(outputs,batch_y)

        loss.backward()

        optimizer.step()

torch.save(net.state_dict(),'saved_model3.pth')
