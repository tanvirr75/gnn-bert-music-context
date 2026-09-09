import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
        self.dropout = dropout
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = torch.relu(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return global_mean_pool(h, batch)


class GNNTagClassifier(nn.Module):
    def __init__(self, encoder, num_tags=50):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_tags)

    def forward(self, x, edge_index, batch):
        return self.classifier(self.encoder(x, edge_index, batch))


class GNNGenreClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, x, edge_index, batch):
        return self.classifier(self.encoder(x, edge_index, batch))