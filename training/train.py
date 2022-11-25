import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import random
# from ptflops import get_model_complexity_info
from sklearn.metrics import precision_score, f1_score, recall_score
sys.path.append("..")
from models.resnet1D import ResNet, BasicBlock
from framework.ours import SDDist_Client, SDDist_Server
from utils.utils import GPU_info, print_label_stat, PAD
import warnings
warnings.filterwarnings('ignore')
# from models.squeezenet import SqueezeNet
import pickle
import time
import sys
import copy


cuda_switch = 1
num_classes = 2
device_num = 28

ref_size = 3200
ref_batch_size = ref_size//2
test_ratio = 0.2

my_seed = 1
train_batch_size = 512
save_path = './checkpoint'
data_path = '../data'

torch.manual_seed(my_seed)
if cuda_switch == 1:
    torch.cuda.manual_seed(my_seed)

ref_record = PAD(data_path, PatientList=range(28, 35), Normalization=True, shuffle=True)

ref_set = Subset(ref_record, range(0,int(ref_size)))
ref_loader = torch.utils.data.DataLoader(ref_set, batch_size=train_batch_size, shuffle=False, num_workers=0)

test_set_2 = Subset(ref_record, range(int(ref_size), len(ref_record)))
test_loader_2 = torch.utils.data.DataLoader(test_set_2, batch_size=train_batch_size, shuffle=False, num_workers=0)

# prepare three datasets (i.e., training set, test set, and validation set) for each device
device_dict = {}
loader_dict = {}
print('Device', end='\t')
for class_id in range(num_classes):
    print(str(class_id), end='\t')
print('SUM')

for device_id in range(device_num):
    patient_record = PAD(data_path, PatientList=[device_id], Normalization=True, shuffle=True)
    train_test_border = int((1-test_ratio)*len(patient_record))
    train_set = Subset(patient_record, range(train_test_border))
#     train_set = Subset(patient_record, range(len(patient_record)))
    
    test_set = Subset(patient_record, range(train_test_border, len(patient_record)))
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=train_batch_size, shuffle=True, num_workers=0)

    loader_dict[device_id] = [train_loader, test_loader]
    print_label_stat(device_id, train_set, num_classes)
    
    
for device_id in range(device_num):
    gpu_id = 0
    device_dict[device_id] = SDDist_Client(device_id, gpu_id=gpu_id, num_classes=num_classes)
    
Fed_server = SDDist_Server(Device_id_list = device_dict.keys(), gpu_id=gpu_id, num_classes=num_classes)


write_log(time.ctime(time.time())+ ' SQMD')
accuracy = 0.
metric = []
for k, v in device_dict.items():
    v.train_local_model_without_ref(1, loader_dict[k][0]) 
#     accuracy += v.validate_model(loader_dict[k][1])
    metric.append(v.validate_model(loader_dict[k][1]))
    v.update_logits(ref_loader)
    GPU_info([v.gpu_id])
metric_arr=np.array(metric)
print('Avg_acc: {:.4}'.format(np.mean(metric_arr, axis=0)[0]))
start_time = time.time()
write_log('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(np.mean(metric_arr, axis=0)[0], 
                                                          np.mean(metric_arr, axis=0)[1],
                                                          np.mean(metric_arr, axis=0)[2],
                                                          np.mean(metric_arr, axis=0)[3],
                                                          time.time()- start_time))

start_time = time.time()
for epoch in range(1000):
    accuracy = 0.
    Fed_server.collect_client_logits(ref_loader, device_dict=device_dict)
    Fed_server.update_top_Q_client(Q=20)
    Fed_server.send_avg_logits_to_clients(K=10, T=3, device_dict=device_dict, method='random')
    metric = []
    for k, v in device_dict.items():
        v.train_local_model_with_ref(ref_set, 1, loader_dict[k][0], ref_batch_size=ref_batch_size, T=3, rho=0.9)
#         accuracy += v.validate_model(loader_dict[k][1])
#         accuracy += v.validate_model(test_loader_2)
        metric.append(v.validate_model(loader_dict[k][1]))
        v.update_logits(ref_loader)
        GPU_info([v.gpu_id])
    metric_arr=np.array(metric)
    log_txt = 'epoch = {:.2f}   Avg_acc: {:.4f}'.format(epoch, np.mean(metric_arr, axis=0)[0])
    print(log_txt)
    print('Avg_acc: {:.4}'.format(accuracy/device_num))
    write_log('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(np.mean(metric_arr, axis=0)[0], 
                                                                np.mean(metric_arr, axis=0)[1],
                                                                np.mean(metric_arr, axis=0)[2],
                                                                np.mean(metric_arr, axis=0)[3],
                                                                time.time()- start_time))
