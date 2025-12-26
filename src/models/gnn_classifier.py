import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCNModel(nn.Module):
    def __init__(self, args: dict):
        super(GCNModel, self).__init__()
        self.num_features = args['num_features']
        self.num_hidden = args['num_hidden']
        self.num_classes = args['num_classes']

        self.conv1 = GCNConv(self.num_features, self.num_hidden)
        self.linear0 = nn.Linear(self.num_features, self.num_hidden)
        self.linear1 = nn.Linear(self.num_hidden * 2, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = gmp(x, batch)

        news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
        news = F.relu(self.linear0(news))
        x = torch.cat([x, news], dim=1)
        x = F.relu(self.linear1(x))

        x = F.log_softmax(self.linear2(x), dim=-1)

        return x
