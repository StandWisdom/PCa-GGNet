import nrrd
import os
import pandas as pd
import numpy as np
import cv2

from PIL import Image
import math
from munch import munchify
import torchvision.transforms as transforms

class DataSet():
    def __init__(self,opts,path='./data/Trainlist-T2.csv',shuffle=True,
                 tumorSlice=True,dataAug=[True,True,False]):
        self.df = pd.read_csv(path)
        if tumorSlice:
            self.df = self.df[self.df['label']!=5]
        if shuffle:
            self.df = self.df.sample(frac=1,replace=False)
        self.sampleNum = self.df.shape[0]
        print('DataSet INFO: {}, Size:{}'.format(list(self.df.columns),self.df.shape))
        self.idx=0

        self.tf_img = TransformImage(opts,random_rotate=dataAug[0],random_vflip=dataAug[1],random_affine=dataAug[2])        
        
    def generateDataBatch(self,idx,batchsize=8):
        i = idx
        if idx == self.sampleNum//batchsize-1:
            df_t = self.df.iloc[i*batchsize:,:]
        else:
            df_t = self.df.iloc[i*batchsize:(i+1)*batchsize,:]
        y = np.array(df_t['label'])

        L_data = []
        for i in range(df_t.shape[0]):
            z = int(df_t.iloc[i,:]['Z'])
            srcpath = df_t.iloc[i,:]['path']

            img_ori = nrrd.read(srcpath)[0].astype(np.int16)
            img = Image.fromarray(img_ori[:,:,z].T)
#            print(srcpath,img_ori.shape)
            
            input_tensor = self.tf_img(img)
            L_data.append(input_tensor.numpy())
        return np.array(L_data),y      
      
    def unsqueeze(self,img):        
        img = img[np.newaxis,:,:,:].astype(np.float32)       
        return img

class TransformImage(object):

    def __init__(self, opts, solid_resize=[320,320], scale=0.45, random_crop=False,
                 random_hflip=False, random_vflip=False,random_rotate=False,random_affine=False,
                 preserve_aspect_ratio=True): #scale=0.875
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std
        

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        print(opts.input_range,opts.mean,opts.std,scale)

        tfs = []
        if solid_resize:            
            width, height = solid_resize
            tfs.append(transforms.Resize((height, width)))

        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)  
            tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())
            
        if random_affine:
            tfs.append(transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fillcolor=66))
                    
        tfs.append(transforms.Grayscale(num_output_channels=3))# should be listed above the random_rotate
        
        if random_rotate:
            tfs.append(transforms.RandomRotation(degrees=(-30,30)))
        
        tfs.append(transforms.ToTensor())
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))
        
        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor