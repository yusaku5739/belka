#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import copy
import gc
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader as tg_DataLoader
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter, scatter_softmax
import torch_geometric as tg
from torch_geometric.data import Data
from torchmetrics.classification import BinaryAccuracy
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from PLA_net.gcn_lib.sparse.torch_edge import DilatedKnnGraph
from PLA_net.gcn_lib.sparse.torch_nn import norm_layer, act_layer, MLP
from torch_geometric.utils import remove_self_loops, add_self_loops

import wandb


# In[2]:


args = {
  "ver": "5",
  "name": "molecule_smile_gcn_data=1:3",
  "seed": 42,
  "soft_label":True,
  "use_all_data": False,
  "use_bb": False,
  "use_whole": True,
  "is_prot": False,
  "concat_embed": False,
  "num_workers": 4,
  "batch_size": 216,
  "feature": "full",
  "add_virtual_node": False,
  "use_gpu": True,
  "device": 0,
  "epochs": 2,
  "lr": 1e-4,
  "dropout": 0.2,
  "num_layers": 20,
  "mlp_layers": 3,
  "hidden_channels": 128,
  "block": "res+",
  "conv": "gen",
  "gcn_aggr": "softmax",
  "norm": "batch",
  "num_tasks": 1,
  "t": 1.0,
  "p": 1.0,
  "learn_t": True,
  "learn_p": False,
  "msg_norm": False,
  "learn_msg_scale": False,
  "conv_encode_edge": True,
  "graph_pooling": "mean",
  "cross_val": 0,
  "task_type": "classification",
  "binary": True,
  "nclasses": 2,
  "num_features": 2,
  "LMPM": False,
  "PLANET": False,
  "use_prot": False,
  "freeze_molecule": True,
  "num_layers_prot": 20,
  "mlp_layers_prot": 3,
  "hidden_channels_prot": 128,
  "msg_norm_prot": False,
  "learn_msg_scale_prot": False,
  "conv_encode_edge_prot": False,
  "use_prot_metadata": False,
  "num_metadata": 240,
  "scalar": False,
  "multi_concat": False,
  "MLP": False,
  "init_adv_training": False,
  "advs": False,
}


# In[3]:


allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
    'posible_explicit_valence': [1, 2, 3, 4, 5, 6, 7, 'misc'],
    'posible_implicit_valence': [1, 2, 3, 4, 5, 6, 7, 'misc']
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list'],
        ]))

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))

def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx, 
    chirality_idx,
    degree_idx,
    formal_charge_idx,
    num_h_idx,
    number_radical_e_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict

def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx, 
    bond_stereo_idx,
    is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
    }

    return feature_dict


# In[4]:


full_atom_feature_dims = get_atom_feature_dims() 
full_bond_feature_dims = get_bond_feature_dims() 

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding 

    


# In[5]:


class GenMessagePassing(tg.nn.MessagePassing):
    def __init__(self, aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 y=0.0, learn_y=False):

        if aggr in ['softmax_sg', 'softmax', 'softmax_sum']:

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_t and (aggr == 'softmax' or aggr == 'softmax_sum'):
                self.learn_t = True
                self.t = torch.nn.Parameter(torch.Tensor([t]), requires_grad=True)
            else:
                self.learn_t = False
                self.t = t

            if aggr == 'softmax_sum':
                self.y = torch.nn.Parameter(torch.Tensor([y]), requires_grad=learn_y)

        elif aggr in ['power', 'power_sum']:

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_p:
                self.p = torch.nn.Parameter(torch.Tensor([p]), requires_grad=True)
            else:
                self.p = p

            if aggr == 'power_sum':
                self.y = torch.nn.Parameter(torch.Tensor([y]), requires_grad=learn_y)
        else:
            super(GenMessagePassing, self).__init__(aggr=aggr)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):

        if self.aggr in ['add', 'mean', 'max', None]:
            return super(GenMessagePassing, self).aggregate(inputs, index, ptr, dim_size)

        elif self.aggr in ['softmax_sg', 'softmax', 'softmax_sum']:

            if self.learn_t:
                out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)
            else:
                with torch.no_grad():
                    out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)

            out = scatter(inputs*out, index, dim=self.node_dim,
                          dim_size=dim_size, reduce='sum')

            if self.aggr == 'softmax_sum':
                self.sigmoid_y = torch.sigmoid(self.y)
                degrees = degree(index, num_nodes=dim_size).unsqueeze(1)
                out = torch.pow(degrees, self.sigmoid_y) * out

            return out


        elif self.aggr in ['power', 'power_sum']:
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            out = torch.pow(out, 1/self.p)

            if self.aggr == 'power_sum':
                self.sigmoid_y = torch.sigmoid(self.y)
                degrees = degree(index, num_nodes=dim_size).unsqueeze(1)
                out = torch.pow(degrees, self.sigmoid_y) * out

            return out

        else:
            raise NotImplementedError('To be implemented')


class MsgNorm(torch.nn.Module):
    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()

        self.msg_scale = torch.nn.Parameter(torch.Tensor([1.0]),
                                            requires_grad=learn_msg_scale)

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg

class GENConv(GenMessagePassing):
    """
     GENeralized Graph Convolution (GENConv): https://arxiv.org/pdf/2006.07739.pdf
     SoftMax  &  PowerMean Aggregation
    """
    def __init__(self, in_dim, emb_dim, 
                 aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 y=0.0, learn_y=False,
                 msg_norm=False, learn_msg_scale=True,
                 encode_edge=False, bond_encoder=False,
                 edge_feat_dim=None,
                 norm='batch', mlp_layers=2,
                 eps=1e-7):

        super(GENConv, self).__init__(aggr=aggr,
                                      t=t, learn_t=learn_t,
                                      p=p, learn_p=learn_p, 
                                      y=y, learn_y=learn_y)

        channels_list = [in_dim]

        for i in range(mlp_layers-1):
            channels_list.append(in_dim*2)

        channels_list.append(emb_dim)

        self.mlp = MLP(channels=channels_list,
                       norm=norm,
                       last_lin=True)

        self.msg_encoder = torch.nn.ReLU()
        self.eps = eps

        self.msg_norm = msg_norm
        self.encode_edge = encode_edge
        self.bond_encoder = bond_encoder
        self.advs = args["advs"]
        if msg_norm:
            self.msg_norm = MsgNorm(learn_msg_scale=learn_msg_scale)
        else:
            self.msg_norm = None

        if self.encode_edge:
            if self.bond_encoder:
                if self.advs:
                    self.edge_encoder = MM_BondEncoder(emb_dim=in_dim)
                else:
                    self.edge_encoder = BondEncoder(emb_dim=in_dim)
            else:
                self.edge_encoder = torch.nn.Linear(edge_feat_dim, in_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = x

        if self.encode_edge and edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = edge_attr

        m = self.propagate(edge_index, x=x, edge_attr=edge_emb)

        if self.msg_norm is not None:
            m = self.msg_norm(x, m)

        h = x + m
        out = self.mlp(h)

        return out

    def message(self, x_j, edge_attr=None):

        if edge_attr is not None:
            msg = x_j + edge_attr
        else:
            msg = x_j

        return self.msg_encoder(msg) + self.eps

    def update(self, aggr_out):
        return aggr_out


class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        super(MRConv, self).__init__()
        self.nn = MLP([in_channels*2, out_channels], act, norm, bias)
        self.aggr = aggr

    def forward(self, x, edge_index):
        """"""
        x_j = tg.utils.scatter_(self.aggr, torch.index_select(x, 0, edge_index[0]) - torch.index_select(x, 0, edge_index[1]), edge_index[1], dim_size=x.shape[0])
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgConv(tg.nn.EdgeConv):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act, norm, bias), aggr)

    def forward(self, x, edge_index):
        return super(EdgConv, self).forward(x, edge_index)


class GATConv(nn.Module):
    """
    Graph Attention Convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels,  act='relu', norm=None, bias=True, heads=8):
        super(GATConv, self).__init__()
        self.gconv = tg.nn.GATConv(in_channels, out_channels, heads, bias=bias)
        m =[]
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, edge_index):
        out = self.unlinear(self.gconv(x, edge_index))
        return out


class SAGEConv(tg.nn.SAGEConv):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be :math:`\ell_2`-normalized. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 norm=True,
                 bias=True,
                 relative=False,
                 **kwargs):
        self.relative = relative
        if norm is not None:
            super(SAGEConv, self).__init__(in_channels, out_channels, True, bias, **kwargs)
        else:
            super(SAGEConv, self).__init__(in_channels, out_channels, False, bias, **kwargs)
        self.nn = nn

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_i, x_j):
        if self.relative:
            x = torch.matmul(x_j - x_i, self.weight)
        else:
            x = torch.matmul(x_j, self.weight)
        return x

    def update(self, aggr_out, x):
        out = self.nn(torch.cat((x, aggr_out), dim=1))
        if self.bias is not None:
            out = out + self.bias
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class RSAGEConv(SAGEConv):
    """
    Residual SAGE convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, relative=False):
        nn = MLP([out_channels + in_channels, out_channels], act, norm, bias)
        super(RSAGEConv, self).__init__(in_channels, out_channels, nn, norm, bias, relative)


class SemiGCNConv(nn.Module):
    """
    SemiGCN convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(SemiGCNConv, self).__init__()
        self.gconv = tg.nn.GCNConv(in_channels, out_channels, bias=bias)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, edge_index):
        out = self.unlinear(self.gconv(x, edge_index))
        return out


class GinConv(tg.nn.GINConv):
    """
    GINConv layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='add'):
        super(GinConv, self).__init__(MLP([in_channels, out_channels], act, norm, bias))

    def forward(self, x, edge_index):
        return super(GinConv, self).forward(x, edge_index)


class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge',
                 act='relu', norm=None, bias=True, heads=8):
        super(GraphConv, self).__init__()
        if conv.lower() == 'edge':
            self.gconv = EdgConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'mr':
            self.gconv = MRConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gat':
            self.gconv = GATConv(in_channels, out_channels//heads, act, norm, bias, heads)
        elif conv.lower() == 'gcn':
            self.gconv = SemiGCNConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gin':
            self.gconv = GinConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'sage':
            self.gconv = RSAGEConv(in_channels, out_channels, act, norm, bias, False)
        elif conv.lower() == 'rsage':
            self.gconv = RSAGEConv(in_channels, out_channels, act, norm, bias, True)
        else:
            raise NotImplementedError('conv {} is not implemented'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, heads=8, **kwargs):
        super(DynConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, heads)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, **kwargs)

    def forward(self, x, batch=None):
        edge_index = self.dilated_knn_graph(x, batch)
        return super(DynConv, self).forward(x, edge_index)


class PlainDynBlock(nn.Module):
    """
    Plain Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, **kwargs):
        super(PlainDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None):
        return self.body(x, batch), batch


class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, **kwargs):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None):
        return self.body(x, batch) + x*self.res_scale, batch


class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels=64, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, **kwargs):
        super(DenseDynBlock, self).__init__()
        self.body = DynConv(in_channels, out_channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)

    def forward(self, x, batch=None):
        dense = self.body(x, batch)
        return torch.cat((x, dense), 1), batch


class ResGraphBlock(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, channels,  conv='edge', act='relu', norm=None, bias=True, heads=8,  res_scale=1):
        super(ResGraphBlock, self).__init__()
        self.body = GraphConv(channels, channels, conv, act, norm, bias, heads)
        self.res_scale = res_scale

    def forward(self, x, edge_index):
        return self.body(x, edge_index) + x*self.res_scale, edge_index


class DenseGraphBlock(nn.Module):
    """
    Dense Static graph convolution block
    """
    def __init__(self, in_channels,  out_channels, conv='edge', act='relu', norm=None, bias=True, heads=8):
        super(DenseGraphBlock, self).__init__()
        self.body = GraphConv(in_channels, out_channels, conv, act, norm, bias, heads)

    def forward(self, x, edge_index):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1), edge_index



# In[6]:


class DeeperGCN(torch.nn.Module):
    def __init__(self, is_prot):
        super(DeeperGCN, self).__init__()

        # Set PM configuration
        if is_prot:
            self.num_layers = args["num_layers_prot"]
            mlp_layers = args["mlp_layers_prot"]
            hidden_channels = args["hidden_channels_prot"]
            self.msg_norm = args["msg_norm_prot"]
            learn_msg_scale = args["learn_msg_scale_prot"]
            self.conv_encode_edge = args["conv_encode_edge_prot"]

        # Set LM configuration
        else:
            self.num_layers = args["num_layers"]
            mlp_layers = args["mlp_layers"]
            hidden_channels = args["hidden_channels"]
            self.msg_norm = args["msg_norm"]
            learn_msg_scale = args["learn_msg_scale"]
            self.conv_encode_edge = args["conv_encode_edge"]

        # Set overall model configuration
        self.dropout = args["dropout"]
        self.block = args["block"]
        self.add_virtual_node = args["add_virtual_node"]
        self.training = True
        self.args = args

        num_classes = args["nclasses"]
        conv = args["conv"]
        aggr = args["gcn_aggr"]
        t = args["t"]
        self.learn_t = args["learn_t"]
        p = args["p"]
        self.learn_p = args["learn_p"]

        norm = args["norm"]

        graph_pooling = args["graph_pooling"]

        # Print model parameters
        print(
            "The number of layers {}".format(self.num_layers),
            "Aggr aggregation method {}".format(aggr),
            "block: {}".format(self.block),
        )
        if self.block == "res+":
            print("LN/BN->ReLU->GraphConv->Res")
        elif self.block == "res":
            print("GraphConv->LN/BN->ReLU->Res")
        elif self.block == "dense":
            raise NotImplementedError("To be implemented")
        elif self.block == "plain":
            print("GraphConv->LN/BN->ReLU")
        else:
            raise Exception("Unknown block Type")

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # Set GCN layer configuration
        for layer in range(self.num_layers):
            gcn = GENConv(
            hidden_channels,
            hidden_channels,
            aggr=aggr,
            t=t,
            learn_t=self.learn_t,
            p=p,
            learn_p=self.learn_p,
            msg_norm=self.msg_norm,
            learn_msg_scale=learn_msg_scale,
            encode_edge=self.conv_encode_edge,
            bond_encoder=True,
            norm=norm,
            mlp_layers=mlp_layers,
            )

            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        # Set embbeding layers
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)


        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        # Set type of pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception("Unknown Pool Type")

        # Set classification layer
        self.graph_pred_linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, input_batch, dropout=True, embeddings=False):

        x = input_batch.x
        edge_index = input_batch.edge_index
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch

        h = self.atom_encoder(x)

        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1)
                .to(edge_index.dtype)
                .to(edge_index.device)
            )
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.block == "res+":

            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                if dropout:
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

                h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = self.norms[self.num_layers - 1](h)
            if dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == "res":

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == "dense":
            raise NotImplementedError("To be implemented")

        elif self.block == "plain":

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = F.relu(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception("Unknown block Type")

        h_graph = self.pool(h, batch)

        if self.args["use_prot"] or embeddings:
            return h_graph
        else:
            return self.graph_pred_linear(h_graph)


# In[7]:


class BELKA_model(torch.nn.Module):
    def __init__(self, is_prot):
        super().__init__()
        self.is_prot = is_prot

        if args["use_whole"]:
            if is_prot:
                self.gcn = DeeperGCN(is_prot=False)
                self.prot_gcn = DeeperGCN(is_prot=True)

                hidden_channel = 64
                if args["concat_embed"]:
                    channels_concat = [128*2, hidden_channel, hidden_channel, 128]
                else:
                    channels_concat = [128, hidden_channel, hidden_channel, 128]
                self.mlp = MLP(channels_concat, norm=args["norm"], last_lin=True)
            else:
            
                self.gcn = DeeperGCN(is_prot=False)

                hidden_channel = 64
                channels_concat = [128, hidden_channel, hidden_channel, 128]
                self.mlp = MLP(channels_concat, norm=args["norm"], last_lin=True)

        self.classifier = torch.nn.Linear(channels_concat[-1], 1)

    def forward(self, x):
        if args["use_whole"]:
            if self.is_prot:
                drug_emb = self.gcn(x[0], embeddings=True)
                prot_emb = self.prot_gcn(x[1], embeddings=True)
                if args["concat_embed"]:
                    encoder_emb = torch.concat([drug_emb, prot_emb])
                else:
                    encoder_emb = drug_emb + prot_emb

            else:
                encoder_emb = self.gcn(x[0], embeddings=True)
            
        final_emb = self.mlp(encoder_emb)

        logit = self.classifier(final_emb)
    
        return logit
    
class BELKA_pl_model(pl.LightningModule):
    
    def __init__(self, protein_name, is_prot=False, prot_graph=None):
        super().__init__()
        self.belka_model = BELKA_model(is_prot)
        self.train_losses = []
        self.valid_losses = []
        self.valid_map = []
        self.valid_out = []
        self.valid_y = []
        self.last_valid_prediction = []
        self.last_valid_y = []
        self.n_epoch = 0
        
        self.accuracy = BinaryAccuracy()
        self.protein_name = protein_name
        self.prot_graph = prot_graph
        self.is_prot = is_prot
        
        self.result = -1
    
    def forward(self, x):
        return self.belka_model(x)
    
    def training_step(self, batch, batch_idx):
        if self.is_prot:
            logit = self.forward([batch, self.prot_graph.to("cuda")])
        else:
            logit = self.forward([batch, self.prot_graph])
        bce_loss = nn.BCEWithLogitsLoss()
        loss = bce_loss(logit.squeeze(), batch.y)

        self.log("train_loss", loss.item(),on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=args["batch_size"])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.is_prot:
            logit = self.forward([batch, self.prot_graph.to("cuda")])
        else:
            logit = self.forward([batch, self.prot_graph])
        out = F.sigmoid(logit.squeeze())
        
        self.log("val_acc", self.accuracy(out, batch.y).item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=args["batch_size"])
        
        self.valid_out.extend(list(out.cpu().to(torch.float32).numpy().flatten()))
        self.valid_y.extend(list(batch.y.cpu().to(torch.float32).numpy().flatten()))
    
    def on_validation_epoch_end(self):
        val_map = average_precision_score(self.valid_y, self.valid_out, average='micro')
        bce_loss = nn.BCELoss()
        val_loss = bce_loss(torch.tensor(np.array(self.valid_out)), torch.tensor(np.array(self.valid_y)))
        self.n_epoch += 1
        self.log("valid_map", val_map,on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("valid_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("epoch", self.n_epoch, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        print(f"epoch{self.n_epoch} valid_loss: {val_loss}   valid_map: {val_map}")

        self.result = max(self.result, val_map)
        
        #self.last_valid_prediction = copy.deepcopy(self.valid_out)
        #self.last_valid_y = copy.deepcopy(self.valid_y)
        
        self.valid_out = []
        self.valid_y = []
        
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return F.softmax(self(batch), dim=1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=args["lr"])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["epochs"], eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_last_valid_prediction(self):
        return self.last_valid_prediction, self.last_valid_y


# In[8]:


"""
class BELKA_dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        
        graph_bb1 = smiles_to_graph(row["buildingblock1_smiles"], is_prot=False, received_mol=False)
        graph_bb2 = smiles_to_graph(row["buildingblock2_smiles"], is_prot=False, received_mol=False)
        graph_bb3 = smiles_to_graph(row["buildingblock3_smiles"], is_prot=False, received_mol=False)
        
        y = row["binds"]
        
        return graph_bb1, graph_bb2, graph_bb3, y
"""
from torch_geometric.data import Dataset as tg_Dataset
from multiprocessing import Pool

class BELKA_whole_dataset(tg_Dataset):
    def __init__(self, df, protein_name, is_prot=False, mode="train"):
        super().__init__()
        self.df = df
        self.is_prot = is_prot
        self.mode=mode
        #self.prot_graph = prot_graph
            #self.mol_dict = {}

        #for i in tqdm(range(len(self.df))):
        #    row = self.df.iloc[i,:]
        #    smile = row["molecule_smiles"]
        #    self.mol_dict[i] = transform_molecule_pg(smile, row["binds"])

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx,:]
        smile = row["molecule_smiles"]

        if self.mode == "test":
            return [transform_molecule_pg(smile, 0), int(row["id"])]
        else:
            target = row["binds"]
        

            if self.mode=="train":
                if args["soft_label"] and row["binds"]:
                    target = 0.8
            

            return transform_molecule_pg(smile, target)
        
       # return self.mol_dict[idx]


# In[9]:


mol_dict = {}

def smiles_to_graph(smiles_string, is_prot=False, received_mol=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    if not is_prot:
        mol = Chem.MolFromSmiles(smiles_string)
    else:
        mol = Chem.MolFromFASTA(smiles_string)
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        ftrs = atom_to_feature_vector(atom)
        atom_features_list.append(ftrs)

    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    return [edge_attr, edge_index, x]

def transform_molecule_pg(smiles,label,received_mol=False,is_prot=False):

    if is_prot:
        edge_attr_p, edge_index_p, x_p = smiles_to_graph(smiles, is_prot)
        x_p = torch.tensor(x_p)
        edge_index_p = torch.tensor(edge_index_p)
        edge_attr_p = torch.tensor(edge_attr_p)

        return Data(edge_attr=edge_attr_p, edge_index=edge_index_p, x=x_p)

    #else:
    
    if args["advs"] or received_mol:
        if args["advs"] or received_mol:
            edge_attr, edge_index, x = smiles_to_graph_advs(
                smiles,
                args,
                advs=True,
                received_mol=received_mol,
                saliency=saliency,
            )
        else:
            edge_attr, edge_index, x = smiles_to_graph_advs(
                smiles, args, received_mol=received_mol, saliency=saliency
            )
    else:
        edge_attr, edge_index, x = smiles_to_graph(smiles)

    #if not saliency:
    x = torch.tensor(x)
    y = torch.tensor([label], dtype=torch.float32)
    edge_index = torch.tensor(edge_index)
    if not args["advs"] and not received_mol:
        edge_attr = torch.tensor(edge_attr)

    if received_mol:
        mol = smiles
    else:
        mol = Chem.MolFromSmiles(smiles)
    
    mol_dict[smiles] = Data(
        edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, mol=mol, smiles=smiles
    )
    
    return mol_dict[smiles]


def make_weights_for_balanced_classes(data, nclasses):
    """
    Generate weights for a balance training loader.
    Args:
        data (list): Labels of each molecule
        nclasses (int): number of classes
    Return:
        weight (list): Weights for each class
    """
    count = [0] * nclasses
    for item in data:
        count[item] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(data)
    for idx, val in enumerate(data):
        weight[idx] = weight_per_class[val]

    return weight



def get_dataloader(fold, prot_name, is_prot, debug):







    
    """
    print("make train_datset")
    df_train_true = df_train.query("binds==1")
    df_train_false = df_train.query("binds==0").sample(n=len(df_train_true), random_state=args["seed"])
    df_train = pd.concat([df_train_true, df_train_false])
    """
    if debug:
        print("load train dataset")
        df_train = pd.read_csv(f"data/fold/{protein_name}/df_train_fold{fold}.csv", usecols=["molecule_smiles", "binds"], nrows=1000)
        df_train_true = df_train.query("binds==1")
        df_train_false = df_train.query("binds==0").sample(n=len(df_train_true), random_state=args["seed"])
        df_train = pd.concat([df_train_true, df_train_false])
        train_dataset = BELKA_whole_dataset(df_train, protein_name=protein_name, is_prot=is_prot)

        """
        df_valid_true = df_valid.query("binds==1")
        df_valid_false = df_valid.query("binds==0").sample(n=len(df_valid_true), random_state=args["seed"])
        df_valid = pd.concat([df_valid_true, df_valid_false])
        if args["use_bb"]:
            valid_dataset = BELKA_dataset(df_valid)
        """
        print("load_valid_data")
        df_valid = pd.read_csv(f"data/fold/{protein_name}/df_valid_fold{fold}.csv", usecols=["molecule_smiles", "binds"], nrows=1000)   
        df_valid_true = df_valid.query("binds==1")
        df_valid_false = df_valid.query("binds==0").sample(n=len(df_valid_true), random_state=args["seed"])
        df_valid = pd.concat([df_valid_true, df_valid_false])
        valid_dataset= BELKA_whole_dataset(df_valid, protein_name=protein_name, is_prot=is_prot)
    else:
        
        
        print("load train dataset")
        if not args["use_all_data"]:
            df_train = pd.read_csv(f"data/fold/{protein_name}/df_train_fold{fold}.csv", usecols=["molecule_smiles", "binds"])
            df_train_true = df_train.query("binds==1")
            df_train_false = df_train.query("binds==0").sample(n=len(df_train_true), random_state=args["seed"])
            df_train = pd.concat([df_train_true, df_train_false])
            train_dataset = BELKA_whole_dataset(df_train, protein_name=protein_name, is_prot=is_prot, mode="train")

            """
            df_valid_true = df_valid.query("binds==1")
            df_valid_false = df_valid.query("binds==0").sample(n=len(df_valid_true), random_state=args["seed"])
            df_valid = pd.concat([df_valid_true, df_valid_false])
            if args["use_bb"]:
                valid_dataset = BELKA_dataset(df_valid)
            """
            print("load_valid_data")
            df_valid = pd.read_csv(f"data/fold/{protein_name}/df_valid_fold{fold}.csv", usecols=["molecule_smiles", "binds"])   
            df_valid_true = df_valid.query("binds==1")
            df_valid_false = df_valid.query("binds==0").sample(n=len(df_valid_true), random_state=args["seed"])
            df_valid = pd.concat([df_valid_true, df_valid_false])
            valid_dataset= BELKA_whole_dataset(df_valid, protein_name=protein_name, is_prot=is_prot, mode="valid")
        else:
            train = get_train(protein_name, debug)

            train["fold"] = -1

            print("prepare df_train, df_valid")
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            for i, (train_index, valid_index) in enumerate(skf.split(train.index, train["binds"])):
                train.iloc[valid_index, -1] = i

            df_train = train.query("fold!=@fold")
            df_valid = train.query("fold==@fold")
            print("make train_datset")
            df_train_true = df_train.query("binds==1")
            df_train_false = df_train.query("binds==0").sample(n=len(df_train_true)*3, random_state=args["seed"])
            df_train = pd.concat([df_train_true, df_train_false])
            
            train_dataset = BELKA_whole_dataset(df_train, protein_name=protein_name, is_prot=is_prot, mode="train")

            print("make valid_dataset")
            df_valid_true = df_valid.query("binds==1")
            df_valid_false = df_valid.query("binds==0").sample(n=len(df_valid_true)*3, random_state=args["seed"])
            df_valid = pd.concat([df_valid_true, df_valid_false])
            valid_dataset= BELKA_whole_dataset(df_valid, protein_name=protein_name, is_prot=is_prot, mode="valid")




    weights_train = make_weights_for_balanced_classes(
            list(df_train["binds"]), 2
        )
    weights_train = torch.DoubleTensor(weights_train)
    sampler_train = torch.utils.data.WeightedRandomSampler(
        weights_train, len(weights_train)
    )
    
    train_loader = tg_DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        sampler=sampler_train,
        pin_memory=True,
        num_workers=args["num_workers"],
    )
    
    valid_loader = tg_DataLoader(
        valid_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=args["num_workers"],
    )
    
    gc.collect()

    return train_loader, valid_loader

"""
def get_dataloader(train, fold):

    train["fold"] = -1
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for i, (train_index, valid_index) in enumerate(skf.split(train.index, train["binds"])):
        train.iloc[valid_index, -1] = i

    df_train = train.query("fold!=@i")
    df_valid = train.query("fold==@i")

    train_dataset = get_dataset(df_train)
    valid_dataset = get_dataset(df_valid)

    weights_train = make_weights_for_balanced_classes(
            list(df_train["binds"]), 2
        )
    weights_train = torch.DoubleTensor(weights_train)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(
        weights_train, len(weights_train)
    )

    train_loader = tg_DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        sampler=sampler_train,
        #pin_memory=True,
        num_workers=args["num_workers"],
    )
    valid_loader = tg_DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        #pin_memory=True,
        num_workers=args["num_workers"],
    )

    return train_loader, valid_loader
"""


# In[10]:


def get_train(protein_name, debug):
    dtypes = {'buildingblock1_smiles': np.int16, 'buildingblock2_smiles': np.int16, 'buildingblock3_smiles': np.int16,
          'binds_BRD4':np.byte, 'binds_HSA':np.byte, 'binds_sEH':np.byte}
    if protein_name in ["sEH", "BRD4", "HSA"]:
        if debug:
            if args["use_bb"]:
                ret = pd.read_csv('data/train.csv', dtype=dtypes, usecols=["molecule_smiles", f"binds_{protein_name}"], nrows=10000)
            elif args["use_whole"]:
                ret = pd.read_csv('data/train.csv', dtype=dtypes, usecols=["molecule_smiles", f"binds_{protein_name}"], nrows=10000)

            ret = ret.rename(columns={f"binds_{protein_name}":"binds"})
            return ret
        else:
            if args["use_bb"]:
                ret = pd.read_csv('data/train.csv', dtype=dtypes, usecols=["molecule_smiles", f"binds_{protein_name}"])
            elif args["use_whole"]:
                ret = pd.read_csv('data/train.csv', dtype=dtypes, usecols=["molecule_smiles", f"binds_{protein_name}"])

            ret = ret.rename(columns={f"binds_{protein_name}":"binds"})
            return ret
    else:
        print(protein_name)
        raise


# In[ ]:

import argparse

import glob

# In[ ]:

def setup_models(protein_name, _ckpt_path, ver):
    ckpt_paths = glob.glob(os.path.join(_ckpt_path, protein_name, str(ver), "*")) 
    models = []
    for ckpt_path in ckpt_paths:
        print(f"load {ckpt_path}")
        models.append(BELKA_pl_model.load_from_checkpoint(ckpt_path, protein_name=protein_name).to("cuda").eval())
        
    return models


if __name__ == "__main__":

    ver=6

    debug = 0

    if debug:
        test = pd.read_csv("data/test.csv", usecols=["id", "molecule_smiles", "protein_name"]).iloc[:600]
    else:
        test = pd.read_csv("data/test.csv", usecols=["id", "molecule_smiles", "protein_name"])
 
    ckpt_path = "output"

    sub_idx = []
    sub_binds = []

    for protein_name in ["sEH", "BRD4", "HSA"]:
        print("#"*30)
        print(f"predict {protein_name}")
        print("#"*30)

        models = setup_models(protein_name=protein_name, _ckpt_path=ckpt_path, ver=ver)
        dataset = BELKA_whole_dataset(test.query("protein_name==@protein_name"), protein_name=protein_name, mode="test")
        dataloader = tg_DataLoader(dataset, shuffle=False, batch_size=args["batch_size"], pin_memory=True,num_workers=args["num_workers"],)

        for x, idx in tqdm(dataloader):

            preds = np.zeros(idx.shape[0])
            for model in models:
                with torch.no_grad():
                    preds += F.sigmoid(model([x.to("cuda"), None])).cpu().numpy().flatten()
            pred = preds/len(models)
            sub_idx.extend(idx.numpy().flatten())
            sub_binds.extend(pred) 
    
    sub = pd.DataFrame({"id":sub_idx, "binds":sub_binds})
    print(sub.head())
    sub.to_csv("submission.csv", index=False)