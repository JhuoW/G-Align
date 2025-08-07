from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Union
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm

def make_conv(in_dim, out_dim, conv_type, **kwargs):
    if conv_type == "GCN":
        return GCNConv(in_dim, out_dim, kwargs['add_self_loops'])
    elif conv_type == "GAT":
        return GATConv(in_dim, out_dim//kwargs['heads'], heads=kwargs['heads'], concat = True)
    elif conv_type == "SAGE":
        return SAGEConv(in_dim, out_dim)
    elif conv_type == "MySAGE":
        return MySAGEConv(in_dim, out_dim, aggr = 'mean', normalize=False, root_weight=True)
    else:
        raise ValueError(f"conv_type {conv_type} not recognized")


class MySAGEConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
            normalize: bool = False,
            root_weight: bool = True,
            project: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = nn.Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = nn.LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = nn.Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Union[Tensor, None] = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # 与PyG-SAGE 不同的是， 这里传入xe作为边特征


        # propagate_type: (x: OptPairTensor, xe: OptTensor)
        out = self.propagate(edge_index=edge_index, size=size, x=x, xe=edge_attr)  # 这里报错
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, xe) -> Tensor:
        return (x_j + xe).relu()  # 邻居节点+边特征作为邻居的聚合特征

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')



class BackboneGNN(nn.Module):
    def __init__(self, in_dim, num_classes, cfg):
        super(BackboneGNN,self).__init__()
        fp_conf = cfg.Fingerprint
        self.n_layers = fp_conf.n_layers
        self.hidden_dim = fp_conf.hidden_dim
        self.gmp = global_mean_pool
        self.gnns = nn.ModuleList()
        self.readout_proj = fp_conf.readout_proj

        self.num_classes = num_classes
        if self.n_layers == 1:  # 1 层
            if not self.readout_proj:
                self.hidden_dim = num_classes
                dims = [in_dim] + [self.hidden_dim]
            else:
                dims = [in_dim] + [self.hidden_dim]
        else:
            if not self.readout_proj:
                dims = [in_dim] + [self.hidden_dim] * (self.n_layers - 1) + [num_classes]
            else:
                dims = [in_dim] + [self.hidden_dim] * (self.n_layers)
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.gnns.append(make_conv(d_in, d_out, fp_conf.conv_type, add_self_loops=fp_conf.add_self_loops, heads=fp_conf.n_heads))
        self.bn = fp_conf.use_bn
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in self.gnns])
        # self.readout_proj = fp_conf.readout_proj
        if self.readout_proj:
            self.proj = nn.Linear(self.hidden_dim, self.num_classes)


    def encode(self, x, edge_index, edge_attr = None):
        h = x
        for l, (conv, bn) in enumerate(zip(self.gnns, self.bns)):
            h = conv(h, edge_index, edge_attr)
            if l != len(self.layers) - 1:
                if self.bn:
                    h = bn(h)
                h = F.relu(h) 
        g = self.gmp(h, edge_index.batch)
        return h, g


    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.xe if hasattr(data, 'xe') else None

        if not self.readout_proj:
            return self.encode(x, edge_index, edge_attr)
        else:
            h, g = self.encode(x, edge_index, edge_attr)
            h = F.relu(h)
            h = self.proj(h)
            return h, g
    

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, act = 'ReLU', layernorm = False):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim) if layernorm else nn.Identity()
        self.act = getattr(F, act.lower(), None)
        self.dropout = dropout
    
    def forward(self, x):
        x = self.lin(x)
        x = self.ln(x)
        if self.act is not None:
            x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
