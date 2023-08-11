import cv2
import numpy as np
import torch
from net_class import Net

def apply_model(path):

    img_size = 100

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size,img_size))
    img_array = np.array(img)

    img_array = img_array / 255
    img_array = torch.tensor(img_array, dtype=torch.float)

    net = Net()
    net.load_state_dict(torch.load('saved_model.pth'))
    net.eval()

    net_out = net(img_array.view(-1, 1, img_size, img_size))[0]
    if net_out[0] >= net_out[1]:
        print('Prediction: BENIGN')
        print(f'Confidence: {round(float(net_out[0]),3)}')
    else:
        print('Prediction: MALIGNANT')
        print(f'Confidence: {round(float(net_out[1]),3)}')



