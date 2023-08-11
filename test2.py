import numpy as np
import torch
from net_class2 import Net

net = Net()
net.load_state_dict(torch.load('saved_model_acceptor.pth'))
net.eval()


testing_data = np.load('acceptor_test_data.npy', allow_pickle=True)

test_X = torch.tensor( [item[0] for item in testing_data], dtype=torch.float )
test_y = torch.tensor( [item[1] for item in testing_data], dtype=torch.float )

correct = 0
total = 0

with torch.no_grad():
    for i in range(len(test_X)):
        output = net(test_X[i].view(-1, 1, 602))[0]
        print(output, test_y[i])
        if output[0] >= output[1]:
            guess ='POSITIVE'
        else:
            guess = 'NEGATIVE'
        
        #print(test_y[i][0])
        real_label = test_y[i][0]
        

        if real_label[0] >= output[1]:
            answer = 'POSITIVE'
        else:
            answer = 'NEGATIVE'
        
        if guess == answer:
            correct += 1

        total +=1

print(f'Correct: {correct}, Total: {total}')
