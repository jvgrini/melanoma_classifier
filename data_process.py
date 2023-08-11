import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


img_size = 100

ben_training_folder ='melanoma_cancer_dataset/train/benign/'
mal_training_folder='melanoma_cancer_dataset/train/malignant/'

ben_test_folder = 'melanoma_cancer_dataset/test/benign/'
mal_test_folder = 'melanoma_cancer_dataset/test/malignant/'

ben_training_data = []
mal_training_data = []

ben_test_data = []
mal_test_data = []

onehot_status = {
    'ben': np.array([1,0]),
    'mal': np.array([0,1])
}

for filename in os.listdir(ben_training_folder):
    try:
    
        path = ben_training_folder+filename
    
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))

        img_array = np.array(img)
        ben_training_data.append([img_array, onehot_status['ben']])
    except: 
        pass    

for filename in os.listdir(mal_training_folder):
    try:
    
        path = mal_training_folder+filename
    
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))

        img_array = np.array(img)
        mal_training_data.append([img_array, onehot_status['mal']])
    except: 
        pass 

for filename in os.listdir(ben_test_folder):
    try:
    
        path = ben_test_folder+filename
    
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))

        img_array = np.array(img)
        ben_test_data.append([img_array, onehot_status['ben']])
    except: 
        pass    

for filename in os.listdir(mal_test_folder):
    try:
    
        path = mal_test_folder+filename
    
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))

        img_array = np.array(img)
        mal_test_data.append([img_array, onehot_status['mal']])
    except: 
        pass 

ben_training_data = ben_training_data[0:len(mal_training_data)]

print(f'Benign training length: {len(ben_training_data)}')
print(f'Mal training length: {len(mal_training_data)}')
print(f'Benign test length: {len(ben_test_data)}')  
print(f'Mal training length: {len(mal_test_data)}')

training_data = np.array(ben_training_data + mal_training_data, dtype=object)
np.random.shuffle(training_data)
np.save('melanoma_training_data.npy', training_data)

test_data = np.array(ben_test_data + mal_test_data, dtype=object)
np.random.shuffle(test_data)
np.save('melanoma_test_data.npy', test_data)