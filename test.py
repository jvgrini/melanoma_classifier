import numpy as np
import torch
from net_class import Net

net = Net()
net.load_state_dict(torch.load('saved_model.pth'))
net.eval()

img_size = 100

testing_data = np.load('melanoma_test_data.npy', allow_pickle=True)

test_X = torch.tensor( [item[0] for item in testing_data] )
test_X = test_X / 255
test_y = torch.tensor( [item[1] for item in testing_data], dtype=torch.float )

correct = 0
total = 0

with torch.no_grad():
    for i in range(len(test_X)):
        output = net(test_X[i].view(-1, 1, img_size, img_size))[0]
        print(output, test_y[i])
        if output[0] >= output[1]:
            guess ='BENIGN'
        else:
            guess = 'MALIGNANT'
        
        real_label = test_y[i]

        if real_label[0] >= output[1]:
            answer = 'BENIGN'
        else:
            answer = 'MALIGNANT'
        
        if guess == answer:
            correct += 1

        total +=1

print(f'Correct: {correct}, Total: {total}')
