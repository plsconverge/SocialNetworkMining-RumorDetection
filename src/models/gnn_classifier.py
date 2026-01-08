import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, DenseSAGEConv, dense_diff_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_scatter import scatter_mean
from math import ceil
import copy as cp


class GCNModel(nn.Module):
    def __init__(self, args: dict, mode='sage', cat=True):
        super(GCNModel, self).__init__()
        self.num_features = args['num_features']
        self.num_hidden = args['num_hidden']
        self.num_classes = args['num_classes']
        self.cat = cat

        if mode == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.num_hidden)
        elif mode == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.num_hidden)
        elif mode == 'gat':
            self.conv1 = GATConv(self.num_features, self.num_hidden)
        else:
            print('Unknown mode, Using SAGE')
            self.conv1 = SAGEConv(self.num_features, self.num_hidden)
        if self.cat:
            self.linear0 = nn.Linear(self.num_features, self.num_hidden)
            self.linear1 = nn.Linear(self.num_hidden * 2, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = gmp(x, batch)

        if self.cat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.linear0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.linear1(x))

        x = F.log_softmax(self.linear2(x), dim=-1)

        return x


class GCNFNModel(nn.Module):
    def __init__(self, args: dict):
        super(GCNFNModel, self).__init__()
        self.num_features = args['num_features']
        self.num_hidden = args['num_hidden']
        self.num_classes = args['num_classes']

        self.conv1 = GATConv(self.num_features, self.num_hidden * 2)
        self.conv2 = GATConv(self.num_hidden * 2, self.num_hidden * 2)
        self.linear0 = nn.Linear(self.num_features, self.num_hidden)
        self.linear1 = nn.Linear(self.num_hidden * 2, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(gmp(x, batch))
        x = F.selu(self.linear1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
        news = F.relu(self.linear0(news))
        x = torch.cat([x, news], dim=1)
        x = F.relu(self.linear1(x))

        x = F.log_softmax(self.linear2(x), dim=-1)
        return x


class GNNforGNNCL(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super(GNNforGNNCL, self).__init__()
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin:
            self.linear = nn.Linear(2 * hidden_channels + out_channels, out_channels)
        else:
            self.linear = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)

        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.linear is not None:
            x = F.relu(self.linear(x))

        return x

class GNNCLModel(nn.Module):
    def __init__(self, in_channels, num_classes, max_nodes):
        super(GNNCLModel, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        hidden_channels = 64
        self.gnn1_pool = GNNforGNNCL(in_channels, hidden_channels, num_nodes)
        self.gnn1_embedding = GNNforGNNCL(in_channels, hidden_channels, hidden_channels, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNNforGNNCL(3 * hidden_channels, hidden_channels, num_nodes)
        self.gnn2_embedding = GNNforGNNCL(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.gnn3_embedding = GNNforGNNCL(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.linear1 = nn.Linear(3 * hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embedding(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embedding(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embedding(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(self.linear2(x), dim=-1), l1 + l2, e1 + e2


class TDrumorGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features + in_features, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = cp.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = cp.copy(x)
        root_idx = data.root_idx
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(root_idx.device)
        batch_size = max(data.batch) + 1

        for num_batch in range(batch_size):
            idx = (torch.eq(data.batch, num_batch))
            root_extend[idx] = x1[root_idx[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.deopout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(root_idx.device)
        for num_batch in range(batch_size):
            idx = (torch.eq(data.batch, num_batch))
            root_extend[idx] = x2[root_idx[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features + in_features, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = cp.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = cp.copy(x)

        root_idx = data.root_idx
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(root_idx.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[root_idx[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(root_idx.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[root_idx[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x

class BiGCNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes):
        super(BiGCNModel, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_features, hidden_features, out_features)
        self.BUrumorGCN = BUrumorGCN(in_features, hidden_features, out_features)
        self.liear = nn.Linear((out_features + hidden_features) * 2, num_classes)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat([TD_x, BU_x], dim=1)
        x = F.log_softmax(self.linear(x), dim=1)
        return x
