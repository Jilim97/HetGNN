import os
from gat_dependency.utils import gat_layer, dense_layer

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GAE
from typing import Optional

from torch import Tensor
from itertools import chain
from collections import OrderedDict
from torch_geometric.data import HeteroData
from torch_geometric.data import Data

    
class DLP_model(nn.Module):
    def __init__(self, heterodata: HeteroData, node_types: list, node_types_to_pred: list, embedding_dim, features_dim: dict,
                 dropout: float=0.2,  **kwargs):
        super().__init__()
        # We learn separate embedding matrices for each node type
        self.node_types = node_types
        self.node_types_to_pred = node_types_to_pred
        self.dropout = dropout

        self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim)
        self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim)
        self.nt1_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[0]].num_nodes,
                                        embedding_dim=embedding_dim)
        self.nt2_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[1]].num_nodes,
                                        embedding_dim=embedding_dim)
        self.nt1_emblin = torch.nn.Linear(features_dim[node_types[0]]+embedding_dim, embedding_dim)
        self.nt2_emblin = torch.nn.Linear(features_dim[node_types[1]]+embedding_dim, embedding_dim)
        self.nt1_nonlin = nn.Sequential(
            torch.nn.Linear(features_dim[node_types[0]], embedding_dim),
            nn.Tanh()
        )
        self.nt2_nonlin = nn.Sequential(
            torch.nn.Linear(features_dim[node_types[1]], embedding_dim),
            nn.Tanh()
        )

        self.lin_layers = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=32, out_features=1),
        )


    def forward(self, data: HeteroData=None, edge_type_label: str=None, 
                    embedding: str='lin', feature: str='dot', return_embeddings: bool=False ) -> Tensor:
        etl = edge_type_label.split(',')
        # x_dict holds the feature matrix of all node_types
        
        if embedding == 'lin':
            x_dict = {self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),
                    self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x)}

        elif embedding == 'nonlin':
            x_dict = {self.node_types[0]: self.nt1_nonlin(data[self.node_types[0]].x),
                    self.node_types[1]: self.nt2_nonlin(data[self.node_types[1]].x)}        
        
        elif embedding == 'emb':
            x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id),
                    self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id)}
        
        elif embedding =='comb':       
            x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id) + self.nt1_lin(data[self.node_types[0]].x),
                    self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id) + self.nt2_lin(data[self.node_types[1]].x)}

        elif embedding =='combrev':       
            x_dict = {self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x) + self.nt1_emb(data[self.node_types[0]].node_id),
                    self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x)+ self.nt2_emb(data[self.node_types[1]].node_id)}

        elif embedding =='comb2':       
            x_dict = {self.node_types[0]: self.nt1_emblin(torch.cat((self.nt1_emb(data[self.node_types[0]].node_id),data[self.node_types[0]].x),dim=1)),
                    self.node_types[1]: self.nt2_emblin(torch.cat((self.nt2_emb(data[self.node_types[1]].node_id),data[self.node_types[1]].x),dim=1))}
                    
        elif embedding =='comb2rev':       
            x_dict = {self.node_types[0]: self.nt1_emblin(torch.cat(data[self.node_types[0]].x,(self.nt1_emb(data[self.node_types[0]].node_id)),dim=1)),
                    self.node_types[1]: self.nt2_emblin(torch.cat(data[self.node_types[1]].x,(self.nt2_emb(data[self.node_types[1]].node_id)),dim=1))}

        elif embedding == 'normal':
            x_dict = {self.node_types[0]: data[self.node_types[0]].x,
                    self.node_types[1]: data[self.node_types[1]].x}
        
        else:
            assert "Wrong embedding method"

        index = data[etl[0], etl[1], etl[2]].edge_label_index

        nt1_faat = x_dict[self.node_types_to_pred[0]]
        nt2_faat = x_dict[self.node_types_to_pred[1]]

        if feature =='dot':
            feat = (nt1_faat[index[0]] * nt2_faat[index[1]])
        elif feature == 'avg':
            feature = ((nt1_faat[index[0]] + nt2_faat[index[1]])/2 )
        elif feature == 'diff':
            feat = (nt1_faat[index[0]] - nt2_faat[index[1]])            
        elif feature == 'sum':
            feat = (nt1_faat[index[0]] - nt2_faat[index[1]])     
        pred = self.lin_layers(feat)
        
        if return_embeddings:
            return pred.squeeze(), x_dict
        else:
            return pred.squeeze()
        
    
class DLP_model2(nn.Module):
    def __init__(self, data: Data, embedding_dim: int=128, dropout: float=0.2,  **kwargs):
        super().__init__()
        # We learn separate embedding matrices for each node type
        self.dropout = dropout

        self.emb = torch.nn.Embedding(num_embeddings=len(data.node_type),
                                        embedding_dim=embedding_dim)

        self.lin_layers = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=32, out_features=1),
        )


    def forward(self, data: Data=None, 
                    feature: str='dot', return_embeddings: bool=False ) -> Tensor:
        # x_dict holds the feature matrix of all node_types

        emb = self.emb(data.node_id)

        index = data.edge_label_index

        feat = emb[index[0]] * emb[index[1]]

        pred = self.lin_layers(feat)

        if return_embeddings:
            return pred.squeeze(), emb
        else:
            return pred.squeeze()
   
