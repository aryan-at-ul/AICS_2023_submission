import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torchvision.io as io
import numpy as np
from IPython.display import Image as dImage
import os
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision import transforms
import torchvision 
import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
# from models import get_feture_extractor_model
from functools import lru_cache


current_file_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torchvision.io as io
import numpy as np
from IPython.display import Image as dImage
import os
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision import transforms
import torchvision 
import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_model(model_name):
#     model = models.resnet18(pretrained=True).to(device)
#     if model_name == 'resnet18':
#         # weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
#         model = models.resnet18(pretrained=True).to(device)
#     elif model_name == 'efficientnet_b0':
#         weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
#         model = models.efficientnet_b0(weights = weights).to(device)
#     else:
#         weights = torchvision.models.DenseNet121_Weights.DEFAULT
#         model = models.densenet121(weights = weights).to(device)

#     # model = models.resnet18(pretrained=True).to(device)
#     # model = models.efficientnet_b0(weights = weights).to(device)
#     # model = torch.load("/home/melkor/projects/img_to_graph/model_densenet_head.pt").to(device)

#     layes_names = get_graph_node_names(model)
#     # print(layes_names)



#     model.eval()
#     feature_extractor = create_feature_extractor(
#         model, return_nodes=['flatten'])

#     return model,feature_extractor



def fet_from_img(img, model, feature_extractor, i = 0):

   # print("reaching here")
#    model, feature_extractor = get_model(model_name)
   mean = [0.485, 0.456, 0.406]
   std = [0.229, 0.224, 0.225]

   # print(img.shape)

   mean = [0.485, 0.485, 0.485]
   std = [0.229, 0.229, 0.229]

   transform_norm = transforms.Compose([transforms.ToTensor(),
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])

   transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
   #  transforms.RandomRotation(20),
   #  transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()
      output =model(img_normalized)
      out = feature_extractor(img_normalized)
      # return out['flatten'],i # this is for densenet and effnet
      return out['flatten'],i#.cpu().detach().numpy().ravel(),i



def fet_img_image(image_path,model):
   img = Image.open(image_path)
   mean = [0.485, 0.456, 0.406]
   std = [0.229, 0.224, 0.225]
   transform_norm = transforms.Compose([transforms.ToTensor(),
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()
      output =model(img_normalized)
      out = feature_extractor(img_normalized)
      return out.cpu().detach().numpy().ravel()


if __name__ == "__main__":
    print("ok")



# @lru_cache(maxsize=100000)
# def get_feture_extractor_model(model_name):
    
#     # auto_transform = weights.transforms()

#     if model_name == 'resnet18':
#         # weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
#         model = models.resnet18(pretrained=True).to(device)
        
#     elif model_name == 'efficientnet_b0':
#         weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
#         model = models.efficientnet_b0(weights = weights).to(device)
        
#     else:
#         weights = torchvision.models.DenseNet121_Weights.DEFAULT
#         model = models.densenet121(weights = weights).to(device)#torch.load(f"{current_file_path}/saved_models/model_densenet_head.pt").to(device)
        
#     layes_names = get_graph_node_names(model)
#     model.eval()
#     feature_extractor = create_feature_extractor(
#         model, return_nodes=['flatten'])
#     # print(layes_names) 
#     return model, feature_extractor
     





# def fet_from_img(img,i = 0,model_name = 'densenet121'):
#     model,feature_extractor = get_feture_extractor_model(model_name)


#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]

#    # print(img.shape)

#     mean = [0.485, 0.485, 0.485]
#     std = [0.229, 0.229, 0.229]

#     transform_norm = transforms.Compose([transforms.ToTensor(),
#     transforms.Resize((224,224)),transforms.Normalize(mean, std)])

#     transform_norm = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((224,224)),
#     transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img_normalized = transform_norm(img).float()
#     img_normalized = img_normalized.unsqueeze_(0)
#     img_normalized = img_normalized.to(device)
#     with torch.no_grad():
#         model.eval()
#         output =model(img_normalized)
#         out = feature_extractor(img_normalized)
#         return out['flatten'],i



