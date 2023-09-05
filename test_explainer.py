import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
# import networkx as nx
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
from sklearn.metrics import confusion_matrix
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import random
# from skorch import NeuralNetClassifier
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transform = T.Compose([
#     T.NormalizeFeatures(),
#     T.ToDevice(device),
#     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True),
# ])
# dataset = Planetoid(path, dataset, transform=transform)
# train_data, val_data, test_data = dataset[0]



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 32)
        self.lin = Linear(32, 2)

    def forward(self, x, edge_index):
        # print(data)
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)




model = GCN(512,256,2)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


model_config = ModelConfig(
    mode='binary_classification',
    task_level='edge',
    return_type='raw',
)

# Explain model output for a single edge:
# edge_label_index = val_data.edge_label_index[:, 0]



from torch_geometric.explain import Explainer, PGExplainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=30, lr=0.003),
    explanation_type='phenomenon',
    edge_mask_type='object',
    node_mask_type='attributes',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    ),
    # Include only the top 10 most important edges:
    # threshold_config=dict(threshold_type=None, value=10),
)

import pickle
loader = None
with open(f'/home/melkor/projects/medical_image_as_graph/saved_data_loader/test_dataloader_10_resnet18.pkl', 'rb') as f:
    loader = pickle.load(f)


loader = DataLoader(loader.dataset, batch_size=1, shuffle=True)

for epoch in range(30):
    for batch in loader:
        # print(type(batch.y))
        batch.y = torch.Tensor(batch.y)
        batch.y = torch.Tensor(torch.flatten(batch.y))
        # batch.y = batch.y.type(torch.LongTensor)
        # print(batch.y)

        loss = explainer.algorithm.train(
            )

explanation = explainer(loader.dataset[0].x, loader.dataset[0].edge_index,target=  torch.Tensor(torch.flatten(torch.Tensor(loader.dataset[0].y).repeat(20, 1))))

print(explanation.available_explanations)
explanation.visualize_feature_importance('feature_importance.png',top_k= 5)

