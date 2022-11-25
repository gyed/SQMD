import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import random
# from ptflops import get_model_complexity_info
from models.resnet1D import ResNet, BasicBlock
from sklearn.metrics import precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')
# from models.squeezenet import SqueezeNet
import pickle
import time
import sys
import copy

def GPU_info(gpu_list):
    for gpu_id in gpu_list:
        print('GPU{}-{:.4f}G  '.format(gpu_id, torch.cuda.memory_allocated(gpu_id)/1024.**3), end='')
    print('')
    
def write_log(log_txt):
    with open("ISGD_log.txt", 'a') as f:
        f.write(log_txt)
        f.write("\n")

def print_label_stat(device_id, subset, num_classes):
    temp_dict = {}
    for class_id in range(num_classes):
        temp_dict[class_id] = 0
    for sample in subset:
        label = sample[1]
        temp_dict[label] += 1
    print(device_id, end='\t')
    for class_id in range(num_classes):
        print(temp_dict[class_id], end='\t')
    print(sum(temp_dict.values()))

    
class PAD(torch.utils.data.Dataset):
    def __init__(self, root, PatientList=[], Normalization = True, shuffle=False):
        super(PAD, self).__init__()
        file_path = root + '/PAD_combine.txt'
        entry = {}
        self.id = PatientList
        self.Normalization = Normalization
        self.data = []
        self.targets = []

        for PatientID in PatientList:
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)[PatientID]
            self.id = PatientID
            self.Normalization = Normalization
            self.data += entry['x'].tolist()
            self.targets += entry['y'].tolist()
        
        if shuffle:
            rand_num = random.randint(0,100)
            random.seed(rand_num)
            random.shuffle(self.data)
            random.seed(rand_num)
            random.shuffle(self.targets)


    def __getitem__(self, index):
        data = self.data[index]
        if self.Normalization:
            _range = np.max(data) - np.min(data)
            data = (data - np.min(data)) / _range
        return torch.from_numpy(data).unsqueeze(0).float(), self.targets[index]

    def __len__(self):
        return len(self.data)