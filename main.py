import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import os
import pdb
import random
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime
import time
import torch.optim as optim

from torch.autograd import Variable


def Rand_Sample(D,K):
    m = D.shape[0]
    random_rows = np.random.choice(m, size=K, replace=False)
    selected_rows = D[random_rows, :]
    selected_rows = np.array(selected_rows, dtype=np.float32)
    return selected_rows,random_rows


def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0

def load_forecast_csv(labelstd,channel, classes,univar=False):
    base_path = '/home/bill/wxl/CoST/datasets/MQTT/data1/'
    pathseg =  [base_path + i for i in ['mqtt_bruteforce', 'normal', 'scan_A', 'scan_sU','sparta']]
    data = []
    label  = []
    for chapath in pathseg: 
        labelsch = 0
        for file_name in os.listdir(chapath):
            if file_name.endswith('.json'):
                file_path = os.path.join(chapath, file_name)
                with open(file_path, 'r') as f:
                    datach = np.array(json.load(f)).reshape(1,-1)
                    data.extend(datach)
                    labelsch += 1
                    if labelsch==labelstd:
                        break
        label.append(labelsch)            
    
    data = np.array(data)
    data_new = []
    total = labelstd - labelstd % channel

    for i in range(classes):
        data_new.append(data[i*total:total*(i+1)])

    data_new = np.array(data_new).reshape(-1,784)

    merged_array = np.array([data_new[i:i+3] for i in range(0, len(data_new), channel)])
    
    np.random.seed(0)

    index_and_value = list(enumerate(merged_array))
    np.random.shuffle(index_and_value)

    index, arr = zip(*index_and_value)

    train_ratio = 0.6 
    test_ratio = 0.2  
    valid_ratio = 0.2 
    

    train_size = int(train_ratio * len(arr))
    test_size = int(test_ratio * len(arr))
    valid_size = int(valid_ratio * len(arr))

    train_slice = np.array(index[:train_size])
    valid_slice = np.array(index[train_size:train_size+test_size])  
    test_slice = np.array(index[train_size+test_size:train_size+test_size+valid_size])  

    # scaler = StandardScaler().fit(merged_array[train_slice])
    # merged_array = scaler.transform(merged_array)

    pred_lens = [24, 48, 96, 288, 672]
    n_covariate_cols = 0
    return merged_array, train_slice, valid_slice, test_slice, pred_lens, n_covariate_cols

import torch
import torch.nn as nn

class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()
        
        # Conv3D layer
        self.conv3d = nn.Conv3d(3, 128, kernel_size=2, stride=2)
        self.conv3d2 = nn.Conv3d(1, 128, kernel_size=2, stride=2)
        # BatchNorm3d layer
        self.bn3d = nn.BatchNorm3d(128)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x):
        x = x.reshape(1,3,2,14,28)
        x = torch.from_numpy(x)
        x = self.dropout(self.relu(self.bn3d(self.conv3d(x))))

        x = self.relu(self.bn3d(self.conv3d2(x.transpose(1,2))))
        x = self.dropout(self.relu(self.bn3d(self.conv3d(x.transpose(1,3)))))
        x = self.relu(self.bn3d(self.conv3d(x.transpose(1,4))))
        return x

class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        
        # Conv3D layer
        self.conv3d = nn.Conv3d(128, 128, kernel_size=2, stride=2)
        
        # BatchNorm3d layer
        self.bn3d = nn.BatchNorm3d(128)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # MaxPool3D layer
        self.maxpool3d = nn.MaxPool3d(kernel_size=2)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,b):
        x = torch.cat((x, b), dim=0)
        x = self.dropout(self.maxpool3d(self.relu(self.bn3d(self.conv3d(x)))))
        x = self.dropout(self.maxpool3d(self.relu(self.bn3d(self.conv3d(x)))))
        x = x.squeeze(dim=2)
        x = x.view(1, -1)
        fc1 = nn.Linear(x.shape[1],64)
        x = fc1(x)
        fc2 = nn.Linear(64,1)
        x = fc2(x)
        x = self.sigmoid(x)
        return x

def cal_metrics(pred, target):
    return {
        ((pred - target) ** 2).mean(),
        np.abs(pred - target).mean()
    }




parser = argparse.ArgumentParser()
parser.add_argument('--labelstd', type = int, default=200, help='每一类样本的个数')
parser.add_argument('--channel', type = int, default=3, help='通道数')
parser.add_argument('--classes', type = int, default=5, help='类别数')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
data, train_slice, valid_slice, test_slice, pred_lens, n_covariate_cols = load_forecast_csv(args.labelstd,args.channel, args.classes)

# pdb.set_trace()
labels = np.array([[i]*(int(len(data)/args.classes)) for i in range(args.classes)]).reshape(-1,1)
K = 20
B = 10
labelstd = int(labels.shape[0]/args.classes)
# pdb.set_trace()
data = data.reshape(len(data),-1)
data_label = np.concatenate((data, labels), axis=1)


#Generate sample set
classes = 0
selected_data = data_label[data_label[:, -1] == classes]
Sa0,line0 = Rand_Sample(selected_data,K)
Sa_label = random.randint(1,args.classes-1)
otr_data = data_label[data_label[:,-1]==Sa_label]
Sa1,line1 = Rand_Sample(otr_data,K)


line1 += Sa_label*labelstd
Sa_con = np.concatenate((Sa0, Sa1), axis=0)
line_con = np.concatenate((line0, line1), axis=0)
#Generate query set

array = np.array([i for i in range(len(data_label))])
remaining_nums = np.setdiff1d(array, line_con)
data_label = data_label[remaining_nums]


selected_data = data_label[data_label[:, -1] == classes]
Q0,Qline0 = Rand_Sample(selected_data,B)
otr_data = data_label[data_label[:,-1]==Sa_label]
Q1,Qline1 = Rand_Sample(otr_data,B)
Q_con = np.concatenate((Q0, Q1), axis=0)




for i in Q_con:
    if int(i[-1])!=0:
        i[-1]=1

for i in Sa_con:
    if int(i[-1])!=0:
        i[-1]=1

J = 0
F_net = FNet()
C_net = CNet()
Sa0 = Sa0[:,:-1]
Sa1 = Sa1[:,:-1]
criterion = nn.MSELoss()
test_slice_oth = np.concatenate((Qline1, line1), axis=0)
test_data_oth = np.array(data[test_slice_oth],dtype=np.float32)
test_slice_0 = np.concatenate((Qline0,line0),axis=0)
test_data_0 = np.array(data[test_slice_0],dtype=np.float32)


x00 = np.array(test_data_0[0],dtype=np.float32)
x01 = np.array(test_data_0[1],dtype=np.float32)
xoth0 = np.array(test_data_oth[0],dtype=np.float32)
xoth1 = np.array(test_data_oth[1],dtype=np.float32)

test_oth_label = np.concatenate((test_data_oth,np.array([1]*len(test_data_oth)).reshape(-1,1)),axis=1)
test_data0_label = np.concatenate((test_data_0,np.array([0]*len(test_data_0)).reshape(-1,1)),axis=1)

test_data_label = np.concatenate((test_oth_label,test_data0_label),axis=0)
np.random.shuffle(test_data_label)
test_data_label = np.array(test_data_label,dtype=np.float32)


# 定义损失函数和优化器

F_net_optimizer = torch.optim.SGD([p for p in F_net.parameters() if p.requires_grad],
                            lr=0.01,
                            momentum=0.9,
                            weight_decay=1e-4)
C_net_optimizer = torch.optim.SGD([p for p in C_net.parameters() if p.requires_grad],
                            lr=0.01,
                            momentum=0.9,
                            weight_decay=1e-4)

criterion = nn.MSELoss()
t = time.time() 
for i in range(1, B):
    print(str(i)+'/'+str(B))
    DeltaScore0 = 0
    DeltaScore1 = 0
    for j in range(1, K):
        A0 = F_net(Sa0[j])
        A1 = F_net(Sa1[j])

        B_star = F_net(Q_con[i][:-1])
        
        # 计算 DeltaScore0 和 DeltaScore1
        DeltaScore0 += C_net(A0, B_star)
        DeltaScore1 += C_net(A1, B_star)
    
    # 计算平均 DeltaScore0 和 DeltaScore1
    DeltaScore0 /= K
    DeltaScore1 /= K
    
    # 根据 DeltaScore0 和 DeltaScore1 进行预测
    if DeltaScore0 < DeltaScore1:
        Predict = torch.tensor(0)
    else:
        Predict = torch.tensor(1)
    
    # 计算预测结果与实际标签的 MSE 损失
    # 计算 MSE 损失
    yiq = torch.tensor(Q_con[i][-1])
    
    loss = criterion(Predict, yiq)
    loss = Variable(loss, requires_grad = True)
    J += loss.item()
    F_net_optimizer.zero_grad()
    C_net_optimizer.zero_grad()
    loss.backward()
    F_net_optimizer.step()
    C_net_optimizer.step()


    # 打印损失函数值 J
    print("Loss J:", J)
t = time.time()  - t
print("训练时间："+str(t))



t = time.time() 
Pre_test = []
Act = []
for x in test_data_label:
    data = x[:-1] 
    label = (x[-1])
    DS0 = (C_net(F_net(data),F_net(x00))+C_net(F_net(data),F_net(x01)))/2
    DSoth = (C_net(F_net(data),F_net(xoth0))+C_net(F_net(data),F_net(xoth1)))/2
    if DS0 > DSoth:
        Pre_test.append(0)
    else:
        Pre_test.append(1)
    Act.append(label)



accuracy = accuracy_score(Act, Pre_test)
precision = precision_score(Act, Pre_test, average='weighted')
recall = recall_score(Act, Pre_test, average='weighted')
f1 = f1_score(Act, Pre_test, average='weighted')
# MSE,MAE = cal_metrics(Act, Pre_test)
eval_res = {
        'Accuracy': accuracy*100,
        'Precision': precision*100,
        'Recall': recall*100,
        'F1 Score': f1*100,
        # 'MSE': MSE,
        # 'MAE':MAE,
        'refer_time': datetime.timedelta(seconds=time.time() - t)
    }
print(eval_res)