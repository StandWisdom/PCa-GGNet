import os,re,sys
import pandas as pd
import numpy as np 
import argparse

import pretrainedmodels
import pretrainedmodels.utils as utils
import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import time

from DataSet import DataSet

parser = argparse.ArgumentParser(description='Generator-net') 
parser.add_argument('--TrainListPath', default='./data/Datalist-T2-bei3-withMask.csv') # Your list for training
parser.add_argument('--ValListPath', default='./data/Datalist-T2-bei3-withMask-val.csv') # Your list for validation
parser.add_argument('--MODELNAME', default='pnasnet5large') # Model type
parser.add_argument('--EPOCHS', default=100)
parser.add_argument('--nb_classes', default=5)
parser.add_argument('--freeze_num', default=-5)
parser.add_argument('--gpu_list', default=[0,1,2])
parser.add_argument('--BATCH_SIZE', default=64)
parser.add_argument('--dstDir', default='./models3')

args = parser.parse_args()
# In[Set Initial Gpu]# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

print(device)
print(torch.cuda.get_device_name(0))
# In[build model]
##print(pretrainedmodels.model_names)
#print('-------------------')
###print(pretrainedmodels.pretrained_settings['pnasnet5large'])
###print(pretrainedmodels.pretrained_settings['inceptionresnetv2'])
##print(pretrainedmodels.pretrained_settings['densenet121'])
##
#model_NameList = ['resnext101_64x4d','alexnet','densenet121','pnasnet5large','inceptionresnetv2']
model_name = args.MODELNAME #
print(pretrainedmodels.pretrained_settings[model_name])
model_info = pretrainedmodels.pretrained_settings[model_name]
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

dim_feats = model.last_linear.in_features # =2048
model.last_linear = nn.Linear(dim_feats, args.nb_classes)

mp = list(model.parameters())
for para in list(model.parameters())[:args.freeze_num]: #15
    para.requires_grad=False 
#for para in list(model.parameters()):
#    para.requires_grad=True

if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model = torch.nn.DataParallel(model, device_ids=args.gpu_list)
model.to(device)
model.train()

## Set your optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-3)
loss_func = nn.CrossEntropyLoss()
#loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1,1]).to(device))

# In[]
def ACC(pred,truth):
    pred_np = pred.argmax(1).cpu().detach().numpy()
    
    truth_np = truth.cpu().numpy()
    acc = accuracy_score(truth_np,pred_np)
    return acc
# In[]
#INPUT_SIZE = list(model_info['imagenet']['input_size'])
BATCH_SIZE = args.BATCH_SIZE
EPOCHS = args.EPOCHS

MODELPATH= os.path.join(args.dstDir,model_name)
if not os.path.exists(MODELPATH):
    os.mkdir(MODELPATH)
# In[main]
dataset = DataSet(model,shuffle=True,tumorSlice=True,path=args.TrainListPath,dataAug=[True,True,True])
valset = DataSet(model,shuffle=False,tumorSlice=True,path=args.ValListPath,dataAug=[False,False,False])

Train_loss=[]
Train_acc=[]
Val_acc=[]
for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    n_batch = dataset.sampleNum//BATCH_SIZE
    for batch_i in range(n_batch):
#    for batch_i in range(1):
        X_train,y_train=dataset.generateDataBatch(batch_i,batchsize=BATCH_SIZE)
#        X = torch.autograd.Variable(X,requires_grad=False)
        X= torch.FloatTensor(X_train).to(device)
        y= torch.LongTensor(y_train).to(device)
        
        pred = model(X)
        loss = loss_func(pred, y)  # for loss
        optimizer.zero_grad()  # clear gradiant
        loss.backward()
        optimizer.step()

        if batch_i % 20 == 0:
            acc = ACC(pred,y)   
            Train_loss.append(float(loss.data.cpu().numpy()))
            print('epoch:{},iter:{},loss:{},ACC={}'.format(epoch,batch_i,loss.data,acc))

    model.eval()
    # Validate Val
    acc = []      
    n_batch = valset.sampleNum//BATCH_SIZE
    for batch_i in range(n_batch//5):
        X_val,y_val=valset.generateDataBatch(batch_i,batchsize=BATCH_SIZE)
#        X = torch.autograd.Variable(X,requires_grad=False)
        X= torch.FloatTensor(X_val).to(device)
        y= torch.LongTensor(y_val).to(device)
        
        pred = model.forward(X)
        acc.append(ACC(pred,y))
#        print(pred.argmax(1))
    acc_value = np.mean(np.array(acc))
    print('epoch:{},Validating,ACC={}'.format(epoch,acc_value))
    Val_acc.append(acc_value)
    
    # Validate Train
    acc_val = []      
    n_batch = dataset.sampleNum//BATCH_SIZE
    for batch_i in range(0,n_batch//5):
        X_val,y_val=dataset.generateDataBatch(batch_i,batchsize=BATCH_SIZE)
        X= torch.FloatTensor(X_val).to(device)
        y= torch.LongTensor(y_val).to(device)
        
        pred = model.forward(X)
        acc_val.append(ACC(pred,y))
    acc_value_train = np.mean(np.array(acc_val))
    print('epoch:{},Training,ACC={}'.format(epoch,acc_value_train))
    Train_acc.append(acc_value_train)
    
    end = time.time()
    print('Running time:{}s.'.format(round(end-start,4)))
    # Save model
    if Val_acc[-1]>np.max(Val_acc[:-1]) and Val_acc[-1]>0.60:
        model_str = model_name+'_'+str(round(acc_value_train,3))+'-'+str(round(Val_acc[-1],3))+\
                    '_'+str(epoch)+'%'+str(EPOCHS)+'-'+str(BATCH_SIZE)
        os.makedirs(os.path.join(MODELPATH,model_str))
        # Save    
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),},
                    os.path.join(MODELPATH,model_str,model_str+'.pt'))    
        df_acc = pd.DataFrame([],columns=['ACC_train','ACC_val'])    
        df_acc['ACC_train']=Train_acc
        df_acc['ACC_val']=Val_acc
        df_acc.to_csv(os.path.join(MODELPATH,model_str,model_str+'.csv'))
#
#    
#    
    
    