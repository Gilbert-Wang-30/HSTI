import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """Graph Convolution layer that applies linear transform and aggregates neighbor info."""
    def __init__(self, in_channels, out_channels, num_relations=1, bias=True):
        super(GraphConv, self).__init__()
        # We expand output channels by num_relations (K) for separate neighbor groups
        self.K = num_relations  # number of adjacency matrices (spatial kernel size)
        # 1x1 convolution (along temporal and node dimensions) to transform features
        self.conv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=(1, 1), bias=bias)
    
    def forward(self, x, A):
        # x: (N, Cin, T, V), A: (K, V, V)
        N, Cin, T, V = x.shape
        assert A.shape[0] == self.K, "Adjacency matrix batch dimension must match K"
        # Linear transformation on input features
        y = self.conv(x)  # shape: (N, out_channels * K, T, V)
        # Reshape to separate the K partitions (if K=1, this is just (N, 1, Cout, T, V))
        N, Cout_K, T, V = y.shape
        Cout = Cout_K // self.K
        y = y.view(N, self.K, Cout, T, V)
        # Aggregate neighbor features using adjacency matrix (einsum for sum_{u in N(v)} y_u * A_{uv})
        # This multiplies the feature of each neighbor (u) by A_{vu} and sums over u
        out = torch.einsum('nkctv, kvw -> nctw', y, A)  # (N, Cout, T, V)
        return out  # no activation here

class STGCNBlock(nn.Module):
    """Spatio-Temporal GCN Block: one spatial graph conv followed by one temporal conv, with residual connection."""
    def __init__(self, in_channels, out_channels, A, kernel_size_t=3, stride=1, dropout=0.0, residual=True):
        super(STGCNBlock, self).__init__()
        self.A = A  # adjacency matrix (tensor of shape K×V×V) used in this block
        self.gconv = GraphConv(in_channels, out_channels, num_relations=A.size(0))
        # Temporal convolution: 1D conv over time (implemented as Conv2d with kernel=(kernel_size_t,1))
        # Use padding to keep time dimension consistent (assuming kernel_size_t is odd for symmetry)
        pad = (kernel_size_t // 2, 0)  # temporal padding, no padding in vertex dimension
        self.tconv = nn.Sequential(
            nn.BatchNorm2d(out_channels),           # normalize across batch, channels, time, nodes
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size_t, 1),
                      stride=(stride, 1), padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )
        # Residual connection
        if not residual:
            # If no residual, use zero (drop connection)
            self.res_connection = lambda x: 0
        elif in_channels != out_channels or stride != 1:
            # If dimensions differ or time dimension is downsampled, use a projection to match shapes
            self.res_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Identity residual (shallow copy of x)
            self.res_connection = nn.Identity()
        self.act = nn.ReLU(inplace=True)  # final activation

    def forward(self, x):
        # Input x: (N, in_channels, T, V)
        res = self.res_connection(x)            # compute residual branch
        x = self.gconv(x, self.A)               # spatial graph convolution
        x = self.tconv(x)                       # temporal convolution (+ BN, ReLU, Dropout)
        x = x + res                             # add residual connection
        x = self.act(x)                         # apply final ReLU
        return x

class STGCN(nn.Module):
    """Full ST-GCN model composed of multiple STGCN blocks, for regression output."""
    def __init__(self, num_nodes=17, in_channels=1, hidden_channels=[16, 32, 64], dropout=0.0, adjacency_matrix=None):
        super(STGCN, self).__init__()
        assert adjacency_matrix is not None, "Adjacency matrix must be provided"
        A = adjacency_matrix
        # Register A as a buffer (not a learnable parameter, but moves with .to(device))
        self.register_buffer('A', A.clone())
        # Build ST-GCN blocks
        self.st_blocks = nn.ModuleList()
        channels = [in_channels] + list(hidden_channels)
        for i in range(len(channels)-1):
            in_c = channels[i]
            out_c = channels[i+1]
            # For first block, we can disable residual if desired (if in_c != out_c)
            residual = True if i != 0 or in_c == out_c else False
            block = STGCNBlock(in_c, out_c, self.A, kernel_size_t=3, stride=1, dropout=dropout, residual=residual)
            self.st_blocks.append(block)
        # Final prediction layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # pool over time and node dimensions
        self.fc = nn.Linear(channels[-1], 1)  # map to 1 output (RUL)
    
    def forward(self, x):
        # x shape: (N, in_channels, T, V)
        for block in self.st_blocks:
            x = block(x)
        # Global average pool over time and nodes:
        # After pooling, x shape = (N, final_channels, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)    # flatten to (N, final_channels)
        out = self.fc(x)            # (N, 1)
        return out  # output is N×1 (one scalar per sample)
