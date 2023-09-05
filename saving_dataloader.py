from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
import torch.nn.functional as F
import pickle
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import random
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import sys
import argparse


curretn_path = os.getcwd()
path = f"{curretn_path}/chest_xray_graphs"
# super_pixels = 150
# model_name = "efficientnet-b0"
# embed_dim = 128

def load_all_from_one_folder(path,type = 0):
    all_files = os.listdir(path)
    all_data = []
    k = 0
    for one_g in all_files:
        print(one_g)
        name = one_g.split(".")[0]
        try:
            G = nx.read_gpickle(f"{path}/{one_g}")  
            data = from_networkx(G)
            print(data)
        except:
            continue

        if type:
            data.y = [1]
        else:
            data.y = [0]
        k+= 1
        data.x = torch.Tensor([torch.flatten(val).tolist() for val in data.x])
        data.name = name
        all_data.append(data)
    return all_data


def permute_array(array):
    permuted_array = []
    for i in range(len(array)):
        permuted_array.append(array[i])
    return permuted_array

def check_if_a_with_name_exisi(path,name):
    all_files = os.listdir(path)
    if name in all_files:
        return True
    else:
        return False





def dataloader(sp = 100, model_name = "densenet121"):
    """
    load train and test data
    """
    print("loading data")
    train_dataset, test_dataset, val_dataset = None, None, None

    if not check_if_a_with_name_exisi(curretn_path,f'saved_data_loader/train_dataloader_{sp}_{model_name}.pkl'):

        train_normal = load_all_from_one_folder(f"{path}/train/NORMAL")
        train_pneumonia = load_all_from_one_folder(f"{path}/train/PNEUMONIA",1)

        test_normal = load_all_from_one_folder(f"{path}/test/NORMAL")
        test_pneumonia = load_all_from_one_folder(f"{path}/test/PNEUMONIA",1)

        val_normal = load_all_from_one_folder(f"{path}/val/NORMAL")
        val_pneumonia = load_all_from_one_folder(f"{path}/val/PNEUMONIA",1)


        train_data_arr = train_normal + train_pneumonia
        test_data_arr = test_normal + test_pneumonia
        val_data_arr = val_normal + val_pneumonia
        # all_data = permute_array(all_data)
        random.shuffle(train_data_arr)
        random.shuffle(test_data_arr)
        random.shuffle(val_data_arr)
        
        train_dataset = train_data_arr#all_data[:int(len(all_data)*0.8)]
        val_dataset = val_data_arr#all_data[int(len(all_data)*0.8):int(len(all_data)*0.8) + 100]
        test_dataset = test_data_arr#all_data[int(len(all_data)*0.8):]
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,drop_last=True)

        with open(f'train_dataloader_{sp}_{model_name}.pkl', 'wb') as f:
            pickle.dump(train_loader, f)

        with open(f'test_dataloader_{sp}_{model_name}.pkl', 'wb') as f:
            pickle.dump(test_loader, f)
        
        with open(f'val_dataloader_{sp}_{model_name}.pkl', 'wb') as f:
            pickle.dump(val_loader, f)
    else:
        with open(f'saved_data_loader/train_dataloader_{sp}_{model_name}.pkl', 'rb') as f:
            train_loader = pickle.load(f)

        with open(f'/saved_data_loadertest_dataloader_{sp}_{model_name}.pkl', 'rb') as f:
            test_loader = pickle.load(f)
        
        with open(f'saved_data_loader/val_dataloader_{sp}_{model_name}.pkl', 'rb') as f:
            val_loader = pickle.load(f)


    return train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Getting arguments')
    parser.add_argument('--model_name', type=str, default='densenet121', help='Name of the model')
    parser.add_argument('--super_pixels', type=int, default=10, help='Number of superpixels')
    args = parser.parse_args()
    model_name = args.model_name
    super_pixels = args.super_pixels
    train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset = dataloader(super_pixels, model_name)
    print("done")





# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data[0].edge_attr)
#     print()

# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super(GCN, self).__init__()
#         torch.manual_seed(12345)
#         # self.conv1 = GCNConv(512, hidden_channels)
#         # self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         # self.conv3 = GCNConv(hidden_channels, hidden_channels)
#         # self.lin = Linear(hidden_channels, 2)
#         self.conv1 = GCNConv(512, 256)
#         self.conv2 = GCNConv(256, 128)
#         self.conv3 = GCNConv(128,64)
#         self.conv4 = GCNConv(64, 32)
#         self.lin1 = Linear(32, 16)
#         self.lin = Linear(16, 2)

#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings 
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)
#         x = x.relu()
#         x = self.conv4(x, edge_index)

#         # 2. Readout layer
#         #x = JumpingKnowledge(mode = 'cat')(x)
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

#         # 3. Apply a final classifier
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.lin1(x)
#         x = x.relu()
#         x = self.lin(x)
        
#         return x

# model = GCN(hidden_channels=64)
# print(model)


# model = GCN(hidden_channels=64)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001)
# criterion = torch.nn.CrossEntropyLoss()

# def train():
#     model.train()

#     for data in train_loader:  # Iterate in batches over the training dataset.
        
#         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
#         data.y = torch.Tensor(data.y)
#         data.y = torch.Tensor(torch.flatten(data.y))
#         data.y = data.y.type(torch.LongTensor)
#         # print(data.y,"kjhkjdhsfkjhsdkjfhksjdhfkjsdhkjfhsdkjhfjs")
#         # print(out,"dsjflkdsjlfkjsdlkfjlkdsjflksdjlfkjsdlkjlkj")
#         loss = criterion(out, data.y)
#         #print(loss.item())
#         #loss = nn.BCELoss(out,data.y)
#         #loss = F.nll_loss(out, data.y)
#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.

# def test(loader):
#      model.eval()

#      correct = 0
#      for data in loader:  # Iterate in batches over the training/test dataset.
#          out = model(data.x, data.edge_index, data.batch)  
#          data.y = torch.Tensor(data.y)
#         #  print("==="*10)
#         #  print(data)
#          pred = out.argmax(dim=1).view(-1,1)  # Use the class with highest probability.
#         #  print(pred,"pred here",data.y)
#          correct += int((pred == data.y).sum())  # Check against ground-truth labels.
#          acc = correct / len(loader.dataset)
#          if acc > 0.91:
#              torch.save(model.state_dict(), 'model_res_10sp.pt')
#      return correct / len(loader.dataset)  # Derive ratio of correct predictions.


# # for epoch in range(1, 11):
# #     train()
# #     try:
# #         train_acc = test(train_loader)
# #         test_acc = test(test_loader)
# #         print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
# #     except Exception as e:
# #         print("error",e)
# #         pass

# # print("number of paramteres for this model",sum(p.numel() for p in model.parameters()))
