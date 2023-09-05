from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import Sequential  as Seq, GCNConv, JumpingKnowledge

import torch
from torch.nn import Sequential as Seq, Linear, ReLU,BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import random
from torch_geometric.nn import GATConv
from skorch import NeuralNetClassifier
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential as seq, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool,GCNConv,GraphConv,TopKPooling,TopKPooling,DynamicEdgeConv,global_max_pool
from torch_geometric.nn import GCNConv, GATv2Conv,GINConv

try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    os.system("pip install -q torchinfo")
    from torchinfo import summary
from utils import load_best_model


curretn_path = os.getcwd()
path = f"{curretn_path}/chest_xray_graphs_50sp_dnsnet121"
path = "/home/melkor/projects/img_to_graph/graph_folder/chest_xray_graphs_10sp_densenet"

embed_dim = 128
X,Y = [],[]
def load_all_from_one_folder(path,type = 0,train_test = 0):
    all_files = os.listdir(path)
    all_data = []
    k = 0
    # if type == 1:# and train_test == 1:
    #      all_files = all_files[0:1201] 
    # if type == 1 and train_test == 1:
    #     all_files = np.random.choice(all_files, size=2600, replace=False)#all_files[0:1301] 

    # if type == 0 and train_test == 1:
    #     more_files = np.random.choice(all_files, size=5000, replace=True)
    #     all_files = np.concatenate((all_files,more_files),axis=0)

         
    for one_g in all_files:
        print(one_g)
        name = one_g.split(".")[0]
        try:
            G = nx.read_gpickle(f"{path}/{one_g}")  #map_location=torch.device('cpu')
            #G = nx.read_gpickle(torch.load(f"{path}/{one_g}",map_location=torch.device('cpu')))
            # print(G.nodes[0]['x'].shape)
            data = from_networkx(G)
            print(data)
        except:
            continue
        yy = [0]
        if type:
            data.y = [1]
            yy = [1]
        else:
            data.y = [0]
        k+= 1
        # print(data.x.shape)
        data.x = torch.Tensor([torch.flatten(val).tolist() for val in data.x])#nx.get_node_attributes(G,'image')
        data.name = name
        # data.x = data.x.type(torch.LongTensor)
        print(k,data)
        X.append([data])
        Y.append(yy[0])
        all_data.append(data)

    return all_data


def permute_array(array):
    permuted_array = []
    for i in range(len(array)):
        permuted_array.append(array[i])
    return permuted_array



def dataloader():
    """
    load train and test data
    """
    print("loading data")
    train_normal = load_all_from_one_folder(f"{path}/train/NORMAL",0,1)
    train_pneumonia = load_all_from_one_folder(f"{path}/train/PNEUMONIA",1,1)

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
    
    #if True:
    #    transform = T.GDC(
    #    self_loop_weight=1,
    #    normalization_in='sym',
    #    normalization_out='col',
    #    diffusion_kwargs=dict(method='ppr', alpha=0.05),
    #    sparsification_kwargs=dict(method='topk', k=128, dim=0),
    #    exact=True,
    #    )
    #    data = transform(val_data_arr[0])


    train_dataset = train_data_arr#all_data[:int(len(all_data)*0.8)]
    val_dataset = val_data_arr#all_data[int(len(all_data)*0.8):int(len(all_data)*0.8) + 100]
    test_dataset = test_data_arr#all_data[int(len(all_data)*0.8):]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,drop_last=True)

    return train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset



train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset = dataloader()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data[0].edge_attr)
    print()

# model = GCN(hidden_channels=64)
# print(model)


# model = GCN(hidden_channels=64)
#model.load_state_dict(torch.load(PATH))


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

        # self.conv1 = GCNConv(1024, 512)
        # self.conv2 = GCNConv(512, 256)
        # self.conv3 = GCNConv(256,128)
        # self.conv4 = GCNConv(128, 64)
        # self.lin1 = Linear(64, 32)
        # #self.lin2 = Linear(128,64)
        # self.lin = Linear(32, 2)


class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN2, self).__init__()
        torch.manual_seed(12345)
        # self.conv1 = GCNConv(512, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.lin = Linear(hidden_channels, 2)
        self.conv1 = GCNConv(1024, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256,128)
        self.conv4 = GCNConv(128, 64)
        self.lin1 = Linear(64, 32)
        #self.lin2 = Linear(128,64)
        self.lin = Linear(32, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        #x = JumpingKnowledge(mode = 'cat')(x)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        #x = self.lin2(x)
        #x = x.relu()
        x = self.lin(x)
        
        return x



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(hidden_channels, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256,128)
        self.conv4 = GCNConv(128, 64)
        self.lin1 = Linear(64, 32)
        #self.lin2 = Linear(128,64)
        self.lin = Linear(32, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin(x)
        
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(hidden_channels, self.hid, heads=self.in_head, dropout=0.3)
        self.conv2 = GATConv(self.hid*self.in_head, 32, concat=False,
                             heads=self.out_head, dropout=0.3)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        self.lin1 = Linear(32, 2)

    def forward(self,x, edge_index,batch):
        
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)

        return x



class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self,input_dim ,dim_h = 64):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(input_dim, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        # self.lin1 = Linear(dim_h*3, dim_h*3)
        # self.lin2 = Linear(dim_h*3, 2)
        self.lin1 = Linear(dim_h*3, 64)
        self.lin2 = Linear(64, 2)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return  F.log_softmax(h, dim=1)



def get_gnn_model(gnn_model , input_size, ensemble = False):
    if gnn_model == "GCN":
        model = GCN(input_size)
        return model
    elif gnn_model == "GIN":
        model = GIN(input_size)
        return model
    elif gnn_model == "GAT":
        model = GAT(input_size)
        return model
    # elif ensemble == True:
    #     model = Ensemble(input_size)
    #     return model
    else:
        raise Exception("model not found")






def load_best_model(input_size, path, gcn):
    """
    load the best model
    """
    model  = get_gnn_model(gcn, input_size)
    # model = Ensemble(input_size, cnn_model, gcn)
    model.load_state_dict(torch.load(path))
    return model



class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB, modelC,num_features):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Linear(2*num_features, num_features)
        
    def forward(self, x, edge_index, batch):
        # print(self,x.shape)
        x1 = self.modelA(x, edge_index,batch)
        x2 = self.modelB(x, edge_index, batch)
        x3 = self.modelC(x, edge_index,batch)
        # print(len(x3),x3)#,x2.shape,x3.shape,"the shapes ")
        x3x = x3#torch.cat((x3[0],x3[1]),dim = 1)
        x = torch.cat((x1, x2, x3x), dim=1)
        # h = torch.cat((h1, h2, h3), dim=1)
        x = self.classifier(F.relu(x))
        return x
print("jhjhjhj")


def Ensemble(cnn_model):
    if cnn_model == 'denset121':
        #input_size, path, gcn  
        #GIN_50_efficientnet-b0_best_model.pt
        modelb = load_best_model(1024,f'outputs/GAT_10_densenet121_best_model.pt','GAT')
        modela = load_best_model(1024,f'outputs/GCN_10_densenet121_best_model.pt','GCN')
        modelc = load_best_model(1024,f'outputs/GIN_150_densenet121_best_model.pt','GIN')

    elif cnn_model == 'efficientnet-b0':
        modelb = load_best_model(1280,f'outputs/GAT_10_efficientnet-b0_best_model.pt','GAT')
        modela = load_best_model(1280,f'outputs/GCN_10_efficientnet-b0_best_model.pt','GCN')
        modelc = load_best_model(1280,f'outputs/GIN_10_efficientnet-b0_best_model.pt','GIN')

    else:
        print("loading all models from here")
        modelb = load_best_model(512,f'outputs/GAT_50_resnet18_best_model.pt','GAT')
        modela = load_best_model(512,f'outputs/GCN_10_resnet18_best_model.pt','GCN')
        modelc = load_best_model(512,f'outputs/GIN_10_resnet18_best_model.pt','GIN')

    # return modela, modelb, modelc
    modela.lin = Identity()
    modelb.lin1 = Identity()
    # model3.lin1 = Identity()
    modelc.lin2 = Identity()

    model = EnsembleModel(modela, modelb, modelc, 2)
    model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5, inplace=False), 
    torch.nn.Linear(in_features=128, 
                    out_features=2, # same number of output units as our number of classes
                    bias=True))
                    #.to(device)

    return model


# model1 = GCN(hidden_channels=64)#torch.load('model_densnet_10sp_gcn.pt')
# model2 = GAT3()#torch.load('model_densnet_100sp_gcn.pt')
# model3 = GIN(dim_h=64)#torch.load('model_densnet_1280_gcn.pt')

# model1.load_state_dict(torch.load('saved_models/model_densnet_10sp_gcn.pt'))
# model2.load_state_dict(torch.load('model_gat_100sp.pt'))
# model3.load_state_dict(torch.load('model_gin_100sp.pt'))

# model1.lin = Identity()
# model2.lin1 = Identity()
# # model3.lin1 = Identity()
# model3.lin2 = Identity()




# model = EnsembleModel(model1, model2, model3, 2)

# model.classifier = torch.nn.Sequential(
#     torch.nn.Dropout(p=0.5, inplace=False), 
#     torch.nn.Linear(in_features=176, 
#                     out_features=2, # same number of output units as our number of classes
#                     bias=True))
#                     #.to(device)

# print(model)
model = Ensemble('denset121')

print("this summary with parmans false")
# summary(model=model, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )
print("this summary with parmans true")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        data.y = torch.Tensor(data.y)
        data.y = torch.Tensor(torch.flatten(data.y))
        data.y = data.y.type(torch.LongTensor)
        # print(data.y,"kjhkjdhsfkjhsdkjfhksjdhfkjsdhkjfhsdkjhfjs")
        # print(out,"dsjflkdsjlfkjsdlkfjlkdsjflksdjlfkjsdlkjlkj")
        loss = criterion(out, data.y)
        #print(loss.item())
        #loss = nn.BCELoss(out,data.y)
        #loss = F.nll_loss(out, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
cfm = None
label = []
predication = []
def test(loader, flag = 0):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         data.y = torch.Tensor(data.y)
        #  print("==="*10)
        #  print(data)
         pred = out.argmax(dim=1).view(-1,1)  # Use the class with highest probability.
        #  print(pred,"pred here",data.y)
         cf_matrix = confusion_matrix(data.y,pred)
         global cfm
         cfm = cf_matrix
         if flag:
            print(cfm)
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        #  label.append(data.y.cpu().numpy())
        #  predication.append(pred.cpu().numpy())
        #  print(label,"jhjgjhgjhgjhgjgj\n",predication)
         acc = correct / len(loader.dataset) 
         if flag and acc > 0.90:
            torch.save(model.state_dict(), "model_densnet_ens_gcn.pt")
         if flag:
            print(f"ROCAUC: {roc_auc_score(data.y.cpu().numpy(),pred.cpu().numpy(),average=None)}")
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 21):
    train()
    try:
        train_acc = test(train_loader)
        # train_acc = 0.0
        test_acc = test(test_loader,1)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
    except Exception as e:
        print("error",e)
        pass
print(cfm)
print("number of paramteres for this model",sum(p.numel() for p in model.parameters()))
