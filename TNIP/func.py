import os
import time
import pandas as pd
import numpy as np
import torch
import tqdm
import cv2

def rotate_flip(X,flip,rotate): 
    # rotate in spatial axis : torch.rot90
    # flip in temporal axis : torch.flip
    if flip == 0:
        return torch.rot90(X,k=int(rotate/90),dims=[3,4])
    else:
        if rotate / 90 == 0: # 0,1,2,3
            return X
        elif rotate / 90 == 1: # 3,0,1,2
            return torch.cat([X[:,:,12:16,:,:],X[:,:,0:4,:,:],X[:,:,4:8,:,:],X[:,:,8:12,:,:]],dim=2)
        elif rotate / 90 == 2: # 2,3,0,1
            return torch.cat([X[:,:,8:12,:,:],X[:,:,12:16,:,:],X[:,:,0:4,:,:],X[:,:,4:8,:,:]],dim=2)
        elif rotate / 90 == 3: # 1,2,3,0
            return torch.cat([X[:,:,4:8,:,:],X[:,:,8:12,:,:],X[:,:,12:16,:,:],X[:,:,0:4,:,:]],dim=2)

    
def Nested_Invariance_Pooling(feature, region_factor):

    total_feature = list()
    per_feat = feature.permute(0,1,2,4,5,3)
    tmp = per_feat.reshape(feature.shape[0]*feature.shape[1],-1,feature.shape[3])
    square_root_pooling = torch.nn.LPPool1d(norm_type=2, kernel_size=feature.shape[3], stride=1)
    tmp = square_root_pooling(tmp)

    feature = tmp.reshape(feature.shape[0],feature.shape[1],feature.shape[2],feature.shape[4],feature.shape[5])

    for R in range(feature.shape[0]):
        # Square Root Pooling
        region_features = []
        for rf in region_factor:
            square_root_pooling = torch.nn.LPPool2d(norm_type=2, kernel_size=rf, stride=1)
            pool = square_root_pooling(feature[R])/(rf[0] * rf[1]) 
            pool = pool.view(pool.shape[0],pool.shape[1],-1)
            max_pool,_= torch.max(input=pool,dim=2) # Max pooling per spatial region
            region_features.append(max_pool)
        region_features = torch.stack(region_features)

        # Average Pooling
        mean_feature = torch.mean(input=region_features,dim=0) 
        total_feature.append(mean_feature)
    total_feature = torch.stack(total_feature)

    # Max Pooling
    max_feature,_= torch.max(input=total_feature,dim=0) #(8,512) -> (512)

    return max_feature
    

def L2_normalization(feature):
    feature = torch.nn.functional.normalize(input=feature,p=2.0,dim=1)
    return feature
