import math,os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

def make_batch_mask(n_nodes, device='cpu'):
    max_node = n_nodes.max().item()
    mask = torch.zeros(len(n_nodes), max_node)
    for idx, nx in enumerate(n_nodes):
        mask[idx, :nx] = 1
    return mask.bool().to(device), max_node


def make_batch(n_nodes, device='cpu'):
    x = []
    for idx, ns in enumerate(n_nodes):
        x.extend([idx] * ns)
    return torch.LongTensor(x).to(device)


def to_pad(feat, mask, max_node, batch_size):
    n_heads, model_dim = feat.shape[-2:]
    feat_shape = (batch_size, max_node, n_heads, model_dim)
    new_feat = torch.zeros(feat_shape).to(feat)
    new_feat[mask] = feat
    return new_feat

def gcn_conv(x, edge_index, edge_weight):
    N, H = x.shape[0], x.shape[1]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    gcn_conv_output = []
    if edge_weight is None:
        value = torch.ones_like(row) * d_norm_in * d_norm_out
    else:
        value = edge_weight * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for i in range(x.shape[1]):
        gcn_conv_output.append( matmul(adj, x[:, i]) )  # [N, D]
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1) # [N, H, D]
    return gcn_conv_output

class TransConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads=1,kernel='simple',use_graph=True,use_weight=True,graph_weight=-1):
        '''
        in_channels should equal to out_channels when use_weight is False
        '''
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels*num_heads)
        self.Wq = nn.Linear(in_channels, out_channels*num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels*num_heads)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel
        self.use_graph = use_graph
        self.use_weight = use_weight
        self.graph_weight=graph_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def full_attention(self, qs, ks, vs, kernel, n_nodes):
        '''
        qs: query tensor [N, H, D]
        ks: key tensor [N, H, D]
        vs: value tensor [N, H, D]
        n_nodes: num of nodes per graph [B]

        return output [N, H, D]
        '''
        if kernel=='simple':
            # normalize input
            qs = qs / torch.norm(qs, p=2) # (N, H, D)
            ks = ks / torch.norm(ks, p=2) # (N, H, D)

            device = qs.device
            node_mask, max_node = make_batch_mask(n_nodes, device)
            batch_size, batch = len(n_nodes), make_batch(n_nodes, device)

            q_pad = to_pad(qs, node_mask, max_node, batch_size)  # [B, M, H, D]
            k_pad = to_pad(ks, node_mask, max_node, batch_size)  # [B, M, H, D]
            v_pad = to_pad(vs, node_mask, max_node, batch_size)  # [B, M, H, D]

            kv_pad = torch.einsum('abcd,abce->adce', k_pad, v_pad) # [B, D, H, D]

            (n_heads, v_dim), k_dim = vs.shape[-2:], ks.shape[-1]
            v_sum = torch.zeros((batch_size, n_heads, v_dim)).to(device)
            v_idx = batch.reshape(-1, 1, 1).repeat(1, n_heads, v_dim)
            v_sum.scatter_add_(dim=0, index=v_idx, src=vs)  # [B, H, D]

            numerator = torch.einsum('abcd,adce->abce', q_pad, kv_pad)
            numerator = numerator[node_mask] + v_sum[batch]

            k_sum = torch.zeros((batch_size, n_heads, k_dim)).to(device)
            k_idx = batch.reshape(-1, 1, 1).repeat(1, n_heads, k_dim)
            k_sum.scatter_add_(dim=0, index=k_idx, src=ks)  # [B, H, D]
            denominator = torch.einsum('abcd,acd->abc', q_pad, k_sum)
            denominator = denominator[node_mask] + torch.index_select(
                n_nodes.float(), dim=0, index=batch
            ).unsqueeze(dim=-1)

            attn_output = numerator / denominator.unsqueeze(dim=-1)  # [N, H, D]

        elif kernel=='sigmoid': # equivalent to but faster than simple

            device = qs.device
            node_mask, max_node = make_batch_mask(n_nodes, device)
            batch_size, batch = len(n_nodes), make_batch(n_nodes, device)

            q_pad = to_pad(qs, node_mask, max_node, batch_size)  # [B, M, H, D]
            k_pad = to_pad(ks, node_mask, max_node, batch_size)  # [B, M, H, D]
            v_pad = to_pad(vs, node_mask, max_node, batch_size)  # [B, M, H, D]

            # numerator
            numerator = torch.sigmoid(torch.einsum("abcd,ebcd->aebc", q_pad, k_pad))  # [B, B, M, H]

            # denominator
            all_ones = torch.ones([numerator.shape[1]]).to(ks.device)  #changed
            epsilon = 1e-9  # a small value to avoid dividing 0
            denominator = torch.einsum("aebc,e->abc", numerator, all_ones) + epsilon  # [B, M, H]
            denominator = denominator.unsqueeze(1).repeat(1, numerator.shape[1], 1, 1)  # [B, B, M, H]

            # compute attention and attentive aggregated results
            attention = numerator / denominator # [B, B, M, H]
            attn_output = torch.einsum("aebc,ebcd->abcd", attention, v_pad)  # [B, M, H, D]
            attn_output = attn_output[node_mask]

        else:
            raise ValueError
        
        return attn_output

    def forward(self, query_input, source_input, n_nodes, edge_index=None, edge_weight=None):
        '''
        n_nodes: (B,) B is batch_size, it indicates the number of nodes in each graph.
        in_channels should equal to out_channels when use_weight is False
        '''
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)

        attention_output=self.full_attention(query,key,value,self.kernel,n_nodes)

        if self.use_graph:
            if self.graph_weight>0:
                final_output = (1-self.graph_weight)*attention_output + self.graph_weight*gcn_conv(value, edge_index, edge_weight)
            else:
                final_output = attention_output + gcn_conv(value, edge_index, edge_weight)
        else:
            final_output=attention_output
        final_output = final_output.mean(dim=1)

        return final_output
    
class DIFFormer_v2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, kernel='simple', 
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, graph_weight=-1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConv(hidden_channels, hidden_channels, kernel=kernel, use_graph=use_graph, use_weight=use_weight, graph_weight=graph_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
    
    def forward(self, x, edge_index, n_nodes):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, n_nodes, edge_index)
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.activation(x)
            layer_.append(x)
        
        # output MLP layer
        x_out = self.fcs[-1](x)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        return x_out

