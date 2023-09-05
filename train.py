import torch
from models import *
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
import networkx as nx
import matplotlib.pyplot as plt

# device = 'cpu'

def train_epoch(model,device,dataloader,loss_fn,optimizer):

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001)
    criterion = loss_fn#torch.nn.CrossEntropyLoss()

    train_loss,train_correct=0.0,0
    model.train()
    correct = 0.0
    ite = 0
    for data in dataloader:

        ite += 1
        out = model(data.x, data.edge_index, data.batch) 
        data.y = torch.Tensor(data.y)
        data.y = torch.Tensor(torch.flatten(data.y))
        data.y = data.y.type(torch.LongTensor)
        pred = out.argmax(dim=1).view(-1,1)
        acc = accuracy_score(data.y, pred.cpu())
        loss = criterion(out.to(device), data.y.to(device))
        loss.backward()  
        train_loss+=loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        optimizer.zero_grad() 
        correct += acc



    return train_loss.item()/ite,correct/ite
  
def valid_epoch(model,device,dataloader,loss_fn):

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001)
    criterion = loss_fn#torch.nn.CrossEntropyLoss()
    valid_loss, val_correct = 0.0, 0
    model.eval()
    ite = 0
    for data in dataloader:
        ite += 1
        out = model(data.x, data.edge_index, data.batch) 
        data.y = torch.Tensor(data.y)
        data.y = torch.Tensor(torch.flatten(data.y))
        data.y = data.y.type(torch.LongTensor)
        pred = out.argmax(dim=1).view(-1,1)
        cfm = confusion_matrix(data.y,pred.cpu())
        # print(cfm)
        acc = accuracy_score(data.y, pred.cpu())
        loss = criterion(out.to(device), data.y.to(device))
        spcficity = cfm[0,0]/(cfm[0,0]+cfm[0,1])
        sensitivity = cfm[1,1]/(cfm[1,1]+cfm[1,0])
        # print(f"Specificity: {spcficity} Sensitivity: {sensitivity} Accuracy: {acc} Loss: {loss}")
        # loss.backward() 
        valid_loss+=loss
        # optimizer.step()
        # optimizer.zero_grad() 
        val_correct += acc #int((pred == data.y).sum()) 

    return valid_loss.item()/ite,val_correct/ite, cfm, spcficity, sensitivity, data.y, pred



# def test(model,device,dataloader,loss_fn,cnn_model_name, gnn_model, superpixel_number):
    # model = torch.load(f'outputs/{gnn_model}_{superpixel_number}_{cnn_model_name}_best_model.pth')
