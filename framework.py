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

class SDDist_Client():
    def __init__(self, Device_id, gpu_id=None, num_classes=2):
        super(SDDist_Client, self).__init__()
        self.id = Device_id
        self.num_classes = num_classes
        self.model = ResNet(BasicBlock, num_blocks=[3,3,3], num_demensions=[32,32,32], in_channels=1, num_classes=num_classes) # ResNet8
        self.logits = torch.zeros(1, self.num_classes)
        self.neighbor_logits = torch.zeros(1, self.num_classes)
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            self.model.cuda(self.gpu_id)
            self.neighbor_logits.cuda(self.gpu_id)
        self.neighbor_list = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        
    def train_local_model_without_ref(self, num_iter, local_loader):
        '''Train the local model on the local dataset. '''
        print('\nDevice: '+ str(self.id) + ' local model training')
        self.model.train()
        for iter_idx in range(num_iter):
            for batch_idx, (data, target) in enumerate(local_loader):
                if self.gpu_id is not None:
                    data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
                output = self.model(data)
                self.optimizer.zero_grad()
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
            if iter_idx % 10 == 0:
                print('Epoch:{:3d}\t\tLoss: {:.8f}'.format(iter_idx, loss.item()))
                
                
    def train_local_model_with_ref(self, ref_set, num_iter, local_loader, ref_batch_size=128, 
                                   train_batch_size=128, T=3.0, rho=0.1): 
        '''The main model supervises the seed model based on the reference dataset.'''
        self.model.train()
        if self.neighbor_logits.size(0) == 1:
            print('Warning: train_model before get_neighbor_logits. Operations skiped: ' + str(self.id))
        else:
            for iter_idx in range(num_iter):
                self.optimizer.zero_grad()
                
                for batch_idx, (data, target) in enumerate(local_loader):
                    # local loss
                    if self.gpu_id is not None:
                        data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
                    local_loss = F.cross_entropy(self.model(data), target)
#                     print(local_loss, end='\t')
                    
                    # neighbor loss
                    idxs = torch.randint(len(ref_set), (ref_batch_size,)) # sample one batch from the reference set
                    data, target = [0]*ref_batch_size, [0]*ref_batch_size
                    for k, v in enumerate(idxs):
                        data[k], _ = ref_set[v]
                        target[k] = self.neighbor_logits[v]
                    data = torch.stack(data, dim = 0)
                    target = torch.stack(target, dim = 0)
                    if self.gpu_id is not None:
                        data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
                    output = F.log_softmax(self.model(data)/T, dim=1)
                    target = F.softmax(target.float()/T, dim=1)
                    neighbor_loss = F.kl_div(output, target, reduction='batchmean') 
                    neighbor_loss = neighbor_loss * train_batch_size / ref_batch_size
#                     print(neighbor_loss)
                
                    total_loss  = local_loss * (1-rho) + neighbor_loss * rho
                    total_loss.backward()
                    self.optimizer.step()
                    
                if iter_idx % 50 == 0:
                    print('Device:{} - Epoch: {:3d} \tLoss: {:.6f}'.format(self.id, iter_idx, total_loss.item()))


    def update_logits(self, ref_loader):
        '''Update the logits of the main model on the reference dataset.'''
        self.logits = torch.zeros(1, self.num_classes)
        if self.gpu_id is not None:
            self.logits = self.logits.cuda(self.gpu_id)
        self.model.eval()
        for batch_idx, (data, target) in enumerate(ref_loader):
            if self.gpu_id is not None:
                data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
            output = self.model(data).detach()
            self.logits = torch.cat((self.logits, output), 0)
        self.logits = self.logits[torch.arange(self.logits.size(0))!=0]  # Prune the first line 


    def validate_model(self, test_loader):
        '''Validate main model on test dataset. '''
        self.model.eval()
        test_loss = 0.
        correct = 0.
        accuracy = 0.
        precision = 0.
        recall = 0.
        f1 = 0.
        for data, target in test_loader:
            if self.gpu_id is not None:
                data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
            output = self.model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            precision += precision_score(target.cpu().data, pred.cpu(), average='macro')*len(target.cpu().data)
            recall += recall_score(target.cpu().data, pred.cpu(), average='macro')*len(target.cpu().data)
            f1 += f1_score(target.cpu().data, pred.cpu(), average='macro')*len(target.cpu().data)
            
        len_ = len(test_loader.dataset)
        test_loss /= len_
        print('Client:{:2d} loss:{:.4f}, Acc: {}/{} ({:.4f})'.format(self.id, test_loss,
                                                                     correct, len_,
                                                                     correct / len_))      
        return correct/len_, precision/len_, recall/len_, f1/len_
        
        
class SDDist_Server():
    def __init__(self, Device_id_list, gpu_id=None, num_classes=10):
        super(SDDist_Server, self).__init__()
        self.id_list = Device_id_list
        self.gpu_id = gpu_id
        self.logits_repo = {}
        self.loss_repo = {}
        self.top_Q_client = []
        for client_id in Device_id_list:
            self.logits_repo[client_id] = None
            self.loss_repo[client_id] = None
        self.num_classes = num_classes
        
    def collect_client_logits(self, ref_loader, frac=1.0, device_dict={}):
        m = max(int(frac * len(self.id_list)), 1)
        client_idxs = np.random.choice(range(len(self.id_list)), m, replace=False)
        for client_id in client_idxs:
            client_logits = device_dict[client_id].logits
            loss = self.calculate_ref_loss(client_id, client_logits, ref_loader)
            self.logits_repo[client_id] = client_logits
            self.loss_repo[client_id] = loss
            
    def send_avg_logits_to_clients(self, K, T, device_dict={}, method='ref_loss_with_distance'):
        for client_id in self.id_list:
            client_id = int(client_id)
            neighbors = self.get_K_Neighbor(client_id, K, T, method=method)
            sum_logits = self.logits_repo[neighbors[0]]
            for neighbor in neighbors[1:]:
                sum_logits += self.logits_repo[neighbor]
            avg_logits = sum_logits / len(neighbors)
            device_dict[client_id].neighbor_logits = avg_logits
        
            
    def calculate_ref_loss(self, client_id, client_logits, ref_loader):
        ref_loss = 0.
        correct = 0.
        target = [0]*len(ref_set)
        for idx in range(len(ref_set)):
            _, target[idx] = ref_set[idx]
        target = torch.tensor(target)
        if self.gpu_id is not None:
            target, output = target.cuda(self.gpu_id), client_logits.cuda(self.gpu_id)
        ref_loss += F.cross_entropy(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        ref_loss /= len(ref_loader.dataset)
        print('Server: Device:{:2d} received ref_loss: {:.4f}, Acc: {}/{} ({:.4f})'.format(client_id, ref_loss, correct,
                                                                                  len(ref_loader.dataset),
                                                                                  correct / len(ref_loader.dataset)))      
        return ref_loss
        
    def calculate_distance(self, client_id, target_client_id, T):
        start = F.log_softmax(self.logits_repo[client_id]/T, dim=1)
        end = F.softmax(self.logits_repo[target_client_id]/T, dim=1)
        return F.kl_div(start, end, reduction='batchmean')
        
        
    def update_top_Q_client(self, Q):
        sorted_loss = sorted(list(self.loss_repo.items()), key= lambda x: (x[1], x[0]))
        self.top_Q_client = list(np.array(sorted_loss)[:Q, 0])
        
    
    def get_K_Neighbor(self, client_id, K, T, method='ref_loss_with_distance'):
        distance_list = []
        if method == 'ref_loss_with_distance':
            if self.top_Q_client == []:
                print('Warning: get_K_Neighbor before update_top_Q_client.')
                return []
            else:
                for target_client_id in self.top_Q_client:
                    if client_id == target_client_id:
                        continue
                    else:
                        distance_list.append([target_client_id, Fed_server.calculate_distance(client_id, target_client_id, T).cpu()])
                distance_list = sorted(distance_list, key= lambda x: (x[1], x[0]))
                return list(np.array(distance_list)[:K, 0])
        elif method == 'distance':
            for target_client_id in self.id_list:
                if client_id == target_client_id:
                    continue
                else:
                    distance_list.append([target_client_id, Fed_server.calculate_distance(client_id, target_client_id, T).cpu()])
            distance_list = sorted(distance_list, key= lambda x: (x[1], x[0]))
            return list(np.array(distance_list)[:K, 0])
        elif method == 'ref_loss':
            if self.top_Q_client == []:
                print('Warning: get_K_Neighbor before update_top_Q_client.')
                return []
            else:
                for target_client_id in self.top_Q_client:
                    if client_id == target_client_id:
                        continue
                    else:
                        distance_list.append(target_client_id)
                return distance_list[:K]
        elif method == 'random':
            for target_client_id in self.id_list:
                if client_id == target_client_id:
                    continue
                else:
                    distance_list.append(target_client_id)
            return random.sample(distance_list, K)
            
