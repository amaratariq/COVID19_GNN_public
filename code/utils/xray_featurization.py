import pandas as pd
import numpy as np
import os
from os import path
import pickle as pkl
import sys
from skimage import exposure
from skimage import io
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py


print(torch.__version__)
print(torchvision.__version__)
np.random.seed(47)


trans = transforms.Compose([transforms.Scale(256),
                            transforms.CenterCrop(256),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                           ])



def image_loader(image_name):
    """load image, returns cuda tensor"""

    ## for equalization
    image = io.imread(image_name)
    img_cdf, bin_centers = exposure.cumulative_distribution(image)
    image = np.interp(image, bin_centers, img_cdf)

    image = Image.fromarray(np.uint8(image*255)).convert('RGB')
    image = trans(image).float()

    ## w/o equalization
    #image = Image.open(image_name)
    #image = trans(image.convert('RGB')).float()
    

    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image  

def xray_feature_vector_per_instance(base_file,  out_file):

    df = pd.read_csv(base_file)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # load model from state_dict, model base is densenet121
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(nn.Linear(1024, 14), nn.Sigmoid())   
    model.load_state_dict(
        torch.load('code/utils/model_state_dict',map_location=torch.device('cpu'))
    )
    # drop classifier
    model2 = nn.Sequential(*list(model.children())[:-1])
    print('model loaded')

    model2.to(device)
    
    header_img = 'data/raw/xray/'
    for i,j in df.iterrows():
    
        try: 
            image_path = header_img+df.at[i, 'RADIOLOGY_ID']+'.png'
            img = image_loader(image_path)
        except:
            print('image not found')
            print(df.at[i, 'RADIOLOGY_ID'])
            print()
            continue
        img = img.to(device)
        feats = model2(img)
        feats = feats.cpu().detach().numpy()
        feats[feats<0] = 0
        feats = feats[0,:,:,:].mean(1).mean(1)
        for k in range(1024):
            df.at[i, 'img_'+str(k)] = feats[idx].item()
    df.to_csv(out_file)
    