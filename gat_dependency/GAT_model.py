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

gnn_factory = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GATv2": geom_nn.GATv2Conv,
    "GraphConv": geom_nn.GraphConv,
    "sageconv": geom_nn.SAGEConv,
}


class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, layer_name='GAT', training=True,
                 return_attention_weights=True, **kwargs):
        super().__init__()
        self.training = training
        self.return_attention_weights = return_attention_weights
        self.heads = kwargs.get('heads')
        gnn_layer = gnn_factory[layer_name]

        # Define GAT layers

        self.gnn1 = gnn_layer(in_channels=in_features, out_channels=hidden_features, **kwargs) # add self loops default true
        self.gnn2 = gnn_layer(in_channels=hidden_features, out_channels=hidden_features, **kwargs)
        # self.gnn3 = gnn_layer(in_channels=hidden_features, out_channels=1, **kwargs)

        # Define MLP to perform classifciation
        self.lin1 = nn.Linear(in_features=hidden_features, out_features=out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # layer 1
        x, (_, alpha1) = self.gnn1(x, edge_index, return_attention_weights=self.return_attention_weights)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        # layer 2
        x, (_, alpha2) = self.gnn2(x, edge_index, return_attention_weights=self.return_attention_weights)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        # MLP
        output = self.lin1(x)
        # output = F.sigmoid(x)

        return output, (alpha1, alpha2)


class GNNModel_general(nn.Module):
    def __init__(self, features, num_classes, heads, layer_name='GAT', training=True,
                 act_fn=nn.ReLU(), dropout=0.2, node_calssif=True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.num_layers = len(features)-1
        assert self.num_layers == len(features)-1 == len(heads),"Wrong paramter sizes"
        self.training = training
        self.return_attention_weights = kwargs.get("return_attention_weights", True)
        self.heads = [1] + heads
        gnn_layer = gnn_factory[layer_name]

        # Define GAT layers
        layers = [gat_layer(in_features=features[i]*self.heads[i], hidden_features=features[i+1],
                            act_fn=act_fn, dropout=self.dropout, heads=self.heads[i+1], ix=i)
                  for i in range(self.num_layers)]
        layers = OrderedDict(chain(*[i.items() for i in layers]))
        self.gnn_layers = geom_nn.Sequential('x, edge_index, return_attention_weights', layers)

        # Define MLP to perform classifciation
        self.lin_layers = nn.Sequential(
            nn.Linear(in_features=features[-1]*self.heads[-1], out_features=512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=256, out_features=num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        embeddings = self.gnn_layers(x, edge_index, self.return_attention_weights)

        return embeddings, self.lin_layers(embeddings)


class MLP_baseline_old(nn.Module):
    def __init__(self, N_nodes, emb_dim, hidden_features, dropout,
                 no_exrta_features, pretrained_embs=None, act_fn=nn.ReLU()):
        super().__init__()
        self.hidden_features = hidden_features
        self.embedding = nn.Embedding(num_embeddings=N_nodes, embedding_dim=emb_dim)
        if pretrained_embs is not None:
            self.embedding.weight.data = torch.nn.Parameter(pretrained_embs)
        if hidden_features:
            self.lin1 = nn.Linear(in_features=emb_dim+no_exrta_features, out_features=hidden_features[0])
        # else:
            # self.lin1 = nn.Linear(in_features=emb_dim+no_exrta_features, out_features=hidden_features[0])
            if len(hidden_features) > 1:
                self.lin2 = nn.Linear(in_features=hidden_features[0], out_features=hidden_features[1])

            self.final = nn.Linear(in_features=hidden_features[-1], out_features=1)
        else:
            self.final = nn.Linear(in_features=emb_dim+no_exrta_features, out_features=1)

        self.dropout = nn.Dropout(p=dropout)       
        self.act_fn = act_fn

    def forward(self, x, extra_features=None, return_embs=False):
        # TODO check if features are backproped
        embeddings = self.embedding(x)
        if extra_features is not None:
            x = torch.cat((embeddings, extra_features[x]), dim=1)
        else:
            x = embeddings

        if self.hidden_features:
            x = self.lin1(x)
            x = self.act_fn(x)
            x = self.dropout(x)

            if len(self.hidden_features) > 1:
                x = self.lin2(x)
                x = self.act_fn(x)
                x = self.dropout(x)

        out = self.final(x)

        if return_embs:
            return out, embeddings
        else:
            return out


class MLP_baseline(nn.Module):
    def __init__(self, N_nodes, emb_dim, hidden_features, dropout,
                 no_exrta_features, pretrained_embs=None, act_fn=nn.ReLU()):
        super().__init__()
        self.hidden_features = hidden_features
        self.embedding = nn.Embedding(num_embeddings=N_nodes, embedding_dim=emb_dim)
        if pretrained_embs is not None:
            self.embedding.weight.data = torch.nn.Parameter(pretrained_embs)

        if hidden_features:
            hidden_features.insert(0, emb_dim+no_exrta_features)
            num_layers = len(hidden_features)-1

            layers = [dense_layer(in_features=hidden_features[ix],
                                out_features=hidden_features[ix+1],
                                act_fn=act_fn, dropout=dropout, ix=ix) for ix in range(num_layers)]
            layers.append(OrderedDict([(f'final', nn.Linear(in_features=hidden_features[-1], out_features=1))]))
            layers = OrderedDict(chain(*[i.items() for i in layers]))
            self.mlp_baseline = nn.Sequential(layers)
            
        else:
            layers = [(OrderedDict([(f'final', nn.Linear(in_features=hidden_features[-1], out_features=1))]))]
            self.mlp_baseline = nn.Sequential(layers)

    def forward(self, x, extra_features=None, edge_embed=False):
        # TODO check if features are backproped
        embeddings = self.embedding(x)
        if extra_features is not None:
            x = torch.cat((embeddings, extra_features[x]), dim=1)
        else:
            x = embeddings

        if edge_embed:
            x = x[:, 0, :] * x[:, 1, :] # construct edge embedding

        return self.mlp_baseline(x)
        

class drug_sensitivity_prediction(nn.Module):
    def __init__(self, hidden_features, classes, dropout, act_fn=nn.ReLU()):
        super().__init__()

        num_layers = len(hidden_features)-1
        layers = [dense_layer(in_features=hidden_features[ix],
                              out_features=hidden_features[ix+1],
                              act_fn=act_fn, dropout=dropout, ix=ix) for ix in range(num_layers)]
        
        layers.append(OrderedDict([(f'final', nn.Linear(in_features=hidden_features[-1], out_features=classes))]))

        layers = OrderedDict(chain(*[i.items() for i in layers]))
        self.drug_layers = nn.Sequential(layers)

    def forward(self, x, extra_features=None):

        out = self.drug_layers(x)

        return out



class SL_link_prediction(nn.Module):
    def __init__(self, hidden_features, dropout, N_nodes, emb_dim,
                 pretrained_embs=None, act_fn=nn.ReLU()):
        super().__init__()

        num_layers = len(hidden_features)
        hidden_features.insert(0, emb_dim)

        self.embedding = nn.Embedding(num_embeddings=N_nodes, embedding_dim=emb_dim)
        if pretrained_embs is not None:
            self.embedding.weight.data = torch.nn.Parameter(pretrained_embs)
            
        layers = [dense_layer(in_features=hidden_features[ix],
                              out_features=hidden_features[ix+1],
                              act_fn=act_fn, dropout=dropout, ix=ix) for ix in range(num_layers)]
        
        layers.append(OrderedDict([(f'final', nn.Linear(in_features=hidden_features[-1], out_features=1))]))
        layers = OrderedDict(chain(*[i.items() for i in layers]))
        
        self.sl_layers = nn.Sequential(layers)
    
    def forward(self, x):
        
        x = self.embedding(x)
        x = x[:, 0, :] * x[:, 1, :] # construct edge embedding

        return self.sl_layers(x)


class custom_ReLU(nn.Module):
    def __init__(self, crispr_threshold=-1):
        super().__init__()
        self.crispr_threshold = crispr_threshold
    def forward(self, x):
        return x if x < self.crispr_threshold else 0
    
# ------------------------------------------------------------------------------------------------------

class GCNsimple(nn.Module):
    def __init__(self, hidden_channels: list, layer_name:str, **kwargs) -> None:
        super().__init__()
        gnnlayer = gnn_factory[layer_name]
        self.gnn1 = gnnlayer(in_channels=hidden_channels[0], out_channels=hidden_channels[1], **kwargs)
        self.gnn2 = gnnlayer(in_channels=hidden_channels[1], out_channels=hidden_channels[2], **kwargs)
        # self.gnn3 = gnnlayer(in_channels=hidden_channels[2], out_channels=hidden_channels[3], **kwargs)

        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x = self.gnn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.6)

        # x = F.relu(self.gnn2(x, edge_index))
        # x = F.dropout(x, training=self.training, p=0.2)
        
        # x = F.relu(self.conv3(x, edge_index))
        # x = F.dropout(x, training=self.training, p=0.2)

        x = self.gnn2(x, edge_index)
        return x

class GATsimple(nn.Module):
    def __init__(self, hidden_channels: list, layer_name:str, **kwargs) -> None:
        super().__init__()

        gnnlayer = gnn_factory[layer_name]
        self.gnn1 = gnnlayer(in_channels=hidden_channels[0], out_channels=hidden_channels[1], add_self_loops=False)
        self.gnn2 = gnnlayer(in_channels=hidden_channels[1], out_channels=hidden_channels[2], add_self_loops=False)
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        # x, attw1 = self.gnn1(x=x, edge_index=edge_index, return_attention_weights=True)
        x, (indices, att_w) = self.gnn1(x=x, edge_index=edge_index, return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        x, (indices, att_w) = self.gnn2(x=x, edge_index=edge_index, return_attention_weights=True)
        return x
    
    
class GCNcustom(nn.Module):
    def __init__(self, features: list, layer_name: str='sageconv', heads: list=None,
                 dropout: float=0.2, act_fn: torch.nn.modules.activation=torch.nn.ReLU,
                   **kwargs) -> None:
        super().__init__(**kwargs)

        self.dropout = dropout
        self.num_layers = len(features)-1 # embedding dimension
        if layer_name == 'GAT':
            assert self.num_layers == len(features)-1 == len(heads),"Wrong paramter sizes"
            self.return_attention_weights = kwargs.get("return_attention_weights", True)
            self.heads = [1] + heads
        else:
            self.return_attention_weights = False

        # Define layers
        
        if layer_name == 'GAT':
            layers = [gat_layer(in_features=features[i]*self.heads[i], hidden_features=features[i+1],
                            act_fn=act_fn, dropout=self.dropout, heads=self.heads[i+1], ix=i,
                            layer_name=layer_name, layer=gnn_factory[layer_name], add_self_loops=False)
                  for i in range(self.num_layers)]
        
            layers = OrderedDict(chain(*[i.items() for i in layers]))
            # self.gnn_layers = geom_nn.Sequential('x, edge_index, return_attention_weights', layers)
            self.gnn_layers = geom_nn.Sequential('x, edge_index', layers)
        else:
            layers = [gat_layer(in_features=features[i], hidden_features=features[i+1],
                            act_fn=act_fn, dropout=self.dropout, heads=None, ix=i,
                            layer_name=layer_name, layer=gnn_factory[layer_name])
                  for i in range(self.num_layers)]
        
            layers = OrderedDict(chain(*[i.items() for i in layers]))
            self.gnn_layers = geom_nn.Sequential('x, edge_index', layers)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        embeddings = self.gnn_layers(x, edge_index)

        return embeddings

class LPsimple_classif(nn.Module):
    
    # for now only do dot product so only forward pass
    def forward(self, x_nt1: Tensor, x_nt2: Tensor, edge_label_index: Tensor) -> Tensor:

        edge_feat_nt1 = x_nt1[edge_label_index[0]]
        edge_feat_nt2 = x_nt2[edge_label_index[1]]

        # Apply dot product for final prediction
        return (edge_feat_nt1 * edge_feat_nt2).sum(dim=-1)
    
class LPdeep_classif(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.lin_layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=64, out_features=1))
    def forward(self, x_nt1: Tensor, x_nt2: Tensor, edge_label_index: Tensor) -> Tensor:
        
        edge_feat_nt1 = x_nt1[edge_label_index[0]]
        edge_feat_nt2 = x_nt2[edge_label_index[1]]

        edge_ = (edge_feat_nt1 * edge_feat_nt2)
        
        return self.lin_layers(edge_)
    
class HeteroData_GNNmodel(nn.Module):
    def __init__(self, heterodata: HeteroData, node_types: list, node_types_to_pred: list, embedding_dim, features_dim: dict,
                 gcn_model: str, features: list, layer_name: str='sageconv', heads: list=None,
                 dropout: float=0.2, act_fn: torch.nn.modules.activation=torch.nn.ReLU, lp_model: str='simple', **kwargs):
        super().__init__()
        # We learn separate embedding matrices for each node type
        self.node_types = node_types
        self.node_types_to_pred = node_types_to_pred
        self.gcn_model = gcn_model
        # features.insert(0, embedding_dim)
        # if isinstance(embedding_dim, int):
        #     self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim)
        #     self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim)
        #     self.nt1_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[0]].num_nodes,
        #                                     embedding_dim=embedding_dim)
        #     self.nt2_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[1]].num_nodes,
        #                                     embedding_dim=embedding_dim)
        #     if len(node_types) == 3:
        #         self.nt3_lin = torch.nn.Linear(features_dim[node_types[2]], embedding_dim)
        #         self.nt3_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[2]].num_nodes,
        #                                           embedding_dim=embedding_dim)    
        # elif isinstance(embedding_dim, dict):
        #     self.nt1_lin = torch.nn.Linear(features_dim[node_types[0]], embedding_dim[node_types[0]])
        #     self.nt2_lin = torch.nn.Linear(features_dim[node_types[1]], embedding_dim[node_types[1]])
        #     self.nt1_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[0]].num_nodes,
        #                                     embedding_dim=embedding_dim[node_types[0]])
        #     self.nt2_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[1]].num_nodes,
        #                                     embedding_dim=embedding_dim[node_types[1]])
        #     if len(node_types) == 3:
        #         self.nt3_lin = torch.nn.Linear(features_dim[node_types[2]], embedding_dim[node_types[2]])
        #         self.nt3_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[2]].num_nodes,
        #                                           embedding_dim=embedding_dim[node_types[2]])    
        # else:
        #     TypeError,"Use correct embedding dim type"
        
        # Instantiate homoGNN
        if self.gcn_model == 'simple':
            self.gnn = GCNsimple(hidden_channels=features, layer_name=layer_name, **kwargs)
            self.gnn = geom_nn.to_hetero(self.gnn, metadata=heterodata.metadata())

        elif self.gcn_model == 'gat':
            self.gnn = GATsimple(hidden_channels=features, layer_name=layer_name, **kwargs)
        else:
            self.gnn = GCNcustom(features=features,
                                 layer_name=layer_name, heads=heads,
                                 dropout=dropout, act_fn=act_fn) # CHECK REST OF THIS FUNCTION 
            self.gnn = geom_nn.to_hetero(self.gnn.gnn_layers, metadata=heterodata.metadata())

        # Convert to heteroGNN

        # Apply classif of supervised edges
        if lp_model == 'simple':
            self.classifier = LPsimple_classif()
        else:
            self.classifier = LPdeep_classif(in_features=features[-1])

    def forward(self, data: HeteroData=None, edge_type_label: str=None,
                return_embeddings: bool=False, x_dict: dict=None) -> Tensor:
        etl = edge_type_label.split(',')
        # x_dict holds the feature matrix of all node_types

        if len(self.node_types) == 2:
            # x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id),
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id)}
            # x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id) + self.nt1_lin(data[self.node_types[0]].x),
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id) + self.nt2_lin(data[self.node_types[1]].x)}
            # x_dict = {self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),
            #           self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x)}
            x_dict = {self.node_types[0]: data[self.node_types[0]].x,
                      self.node_types[1]: data[self.node_types[1]].x}
        else:
            # x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id),
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id),
            #           self.node_types[2]: self.nt3_emb(data[self.node_types[2]].node_id)}
            # x_dict = {self.node_types[0]: self.nt1_lin(data[self.node_types[0]].x),
            #           self.node_types[1]: self.nt2_lin(data[self.node_types[1]].x),
            #           self.node_types[2]: self.nt3_lin(data[self.node_types[2]].x)}
            # x_dict = {self.node_types[0]: self.nt1_emb(data[self.node_types[0]].node_id) + self.nt1_lin(data[self.node_types[0]].x),
            #           self.node_types[1]: self.nt2_emb(data[self.node_types[1]].node_id) + self.nt2_lin(data[self.node_types[1]].x),
            #           self.node_types[2]: self.nt3_emb(data[self.node_types[2]].node_id) + self.nt3_lin(data[self.node_types[2]].x)}
            x_dict = {self.node_types[0]: data[self.node_types[0]].x,
                      self.node_types[1]: data[self.node_types[1]].x,
                      self.node_types[2]: data[self.node_types[2]].x}

        if self.gcn_model == 'simple':
            x_dict = self.gnn(x_dict, data.edge_index_dict)
        else:
            x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(x_dict[self.node_types_to_pred[0]],
                                x_dict[self.node_types_to_pred[1]],
                                data[etl[0], etl[1], etl[2]].edge_label_index)
        if pred.ndim == 2:
            pred = pred.ravel()

        if self.gcn_model == 'gat:':
            if return_embeddings:
                return pred, x_dict
            else:
                return pred
        if return_embeddings:
            return pred, x_dict
        else:
            return pred
    


class GAEEncoder(nn.Module):
    def __init__(self, node_types, features, embedding_dim,
                 heterodata, layer_name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.node_types = node_types
        features.insert(0, embedding_dim)
        if isinstance(embedding_dim, int):
            self.nt1_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[0]].num_nodes,
                                            embedding_dim=embedding_dim)
            self.nt2_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[1]].num_nodes,
                                            embedding_dim=embedding_dim)
            self.nt3_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[2]].num_nodes,
                                            embedding_dim=embedding_dim)
        elif isinstance(embedding_dim, dict):
            self.nt1_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[0]].num_nodes,
                                            embedding_dim=embedding_dim[node_types[0]])
            self.nt2_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[1]].num_nodes,
                                            embedding_dim=embedding_dim[node_types[1]])
            self.nt3_emb = torch.nn.Embedding(num_embeddings=heterodata[node_types[2]].num_nodes,
                                            embedding_dim=embedding_dim[node_types[2]])
        else:
            TypeError,"Use correct embedding dim type"

        self.gnn = GCNsimple(hidden_channels=features, layer_name=layer_name, **kwargs)
        self.gnn = geom_nn.to_hetero(self.gnn, metadata=heterodata.metadata())
        
    def forward(self, data: HeteroData) -> Tensor:

        x_dict = {i: emb_i(data[i].node_id) for i, emb_i in zip(self.node_types, [self.nt1_emb, self.nt2_emb, self.nt3_emb])}

        return self.gnn(x=x_dict, edge_index=data.edge_index_dict)


class InnerProductDecoder_hetero(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z: Tensor, edge_index: Tensor, node_type1: str, node_type2: str,
                sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[node_type1][edge_index[0]] * z[node_type2][edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


def recon_loss(self, decoder, z: Tensor, pos_edge_index: Tensor,
               neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # if neg_edge_index is None:
            # neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              1e-15).mean()

        return pos_loss + neg_loss
