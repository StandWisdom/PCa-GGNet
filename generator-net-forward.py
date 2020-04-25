import os
import pandas as pd 
import numpy as np

import pretrainedmodels

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

from DataSet import DataSet
# In[]
Dir = './models'
model_name = '' # could be fbresnet152 or inceptionresnetv2
mymodel_str = ''
PATH = os.path.join(Dir,model_name,mymodel_str,mymodel_str+'.pt')

# In[Set Initial Gpu]
ngpu= torch.cuda.device_count()
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print(device)
print(torch.cuda.get_device_name(0))
# In[build model]

print(pretrainedmodels.pretrained_settings[model_name])
model_info = pretrainedmodels.pretrained_settings[model_name]
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

dim_feats = model.last_linear.in_features # =2048
nb_classes = 5
model.last_linear = nn.Linear(dim_feats, nb_classes)
model.load_state_dict(torch.load(PATH)['model_state_dict'])

mp = list(model.parameters())
for para in list(model.parameters())[:-5]:
    para.requires_grad=False 
#for para in list(model.parameters()):
#    para.requires_grad=True

if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model = torch.nn.DataParallel(model, device_ids=[0,1])

model.to(device)
# In[]
def SoftMax(x):
    y=[]
    for i in range(x.shape[0]):
        x_ = np.exp(x[i])/np.sum(np.exp(x))
        y.append(x_)
    return y


# In[]
#INPUT_SIZE = list(model_info['imagenet']['input_size'])
BATCH_SIZE = 16
preproL = [[100,100],model.input_size,True]
# In[]
dataset = DataSet(model,shuffle=False,tumorSlice=False,path='./data/Datalist-T2-bei3-withMask.csv',dataAug=[False,False,False])

# Validate Train
acc_val = []  
Preds = []    
Scores = []
Fts = []
n_batch = dataset.sampleNum//BATCH_SIZE

model.eval()
for batch_i in range(0,n_batch):
#for batch_i in range(0,1):
    print(batch_i)
    X_,y_=dataset.generateDataBatch(batch_i,batchsize=BATCH_SIZE)
    X= torch.FloatTensor(X_).to(device)
    y= torch.LongTensor(y_).to(device)
    
    if model_name=='pnasnet5large':
        ft,pred = model.myforward(X)
    else:
        pred = model.forward(X)
    # Process pred
    pred_np = pred.argmax(1).cpu().detach().numpy()
    Preds.extend(pred_np)
    pred_array = pred.cpu().detach().numpy()
    for i in range(pred_array.shape[0]):
        Scores.append(SoftMax(pred_array[i,:]))    
    acc_val.append(accuracy_score(pred_np,y.cpu().numpy()))
    
    if model_name=='pnasnet5large':
        # Precess ft
        ft_np = ft.cpu().detach().numpy()
        for i in range(pred_array.shape[0]):
            Fts.append(ft_np[i,:])    
        
acc_value = np.mean(np.array(acc_val))
print('Train ACC={}'.format(acc_value))

df = dataset.df
mydf = pd.DataFrame([])
mydf['ID']=df['ID']
mydf['label']=np.array(df['label'])
mydf['Z']=df['Z']

# For ISUP prob
mydf['pred'] = list(Preds)
scores_array = np.array(Scores)
mydf['ISUP0']=list(scores_array[:,0])
mydf['ISUP1']=list(scores_array[:,1])
mydf['ISUP2']=list(scores_array[:,2])
mydf['ISUP3']=list(scores_array[:,3])
mydf['ISUP4']=list(scores_array[:,4])

if model_name=='pnasnet5large':
    # For fullconection layer outputs
    Fts_array=np.array(Fts)
    for i in range(Fts_array.shape[1]):
        mydf['Ft_'+str(i)]=Fts_array[:,i]

mydf.to_csv(os.path.join(Dir,model_name,mymodel_str,mymodel_str+'-ft.csv'),index=0)

