import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, dataset, num_classes):
        super.__init__()
        self.conv1 = GCNConv(in_channels=dataset.num_node_features,
                             out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # only used in training
        x = self.conv2(x, edge_index)

        return F.sigmoid(x)
