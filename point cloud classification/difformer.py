import math,os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

def to_block(input, n_nodes):
    '''
    input: (N, n_col), n_nodes: (B)
    '''
    feat_list=[]
    cnt=0
    for n in n_nodes:
        feat_list.append(input[cnt:cnt+n])
        cnt+=n
    blocks=torch.block_diag(*feat_list)

    return blocks # (N, n_col*B)
    
def unpack_block(input, n_col, n_nodes):
    '''
    input: (N, B*n_col), n_col: int, n_nodes: (B)
    '''
    feat_list=[]
    cnt=0
    start_col=0
    for n in n_nodes:
        feat_list.append(input[cnt:cnt+n,start_col:start_col+n_col])
        cnt+=n
        start_col+=n_col
    
    return torch.cat(feat_list,dim=0) # (N, n_col)

def batch_repeat(input, n_col, n_nodes):
    '''
    input: (B*n_col), n_col: int, n_nodes: (B)
    '''
    x_list=[]
    cnt=0
    for n in n_nodes:
        x=input[cnt:cnt+n_col].repeat((n,1)) # (n, n_col)
        x_list.append(x)
        cnt+=n_col
    
    return torch.cat(x_list,dim=0)    

def gcn_conv(x, edge_index, edge_weight):
    if edge_weight is None:
        N=x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value=torch.ones_like(row) * d_norm_in * d_norm_out
    else:
        value=edge_weight
    value = torch. nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj=torch.sparse_coo_tensor(edge_index,value,size=(N,N))
    # adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    return torch.sparse.mm(adj, x)

class TransConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel='simple',use_graph=True,use_weight=True,graph_weight=-1):
        '''
        in_channels should equal to out_channels when use_weight is False
        '''
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels)
        self.Wq = nn.Linear(in_channels, out_channels)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
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
        
        '''
        if kernel=='simple':
            # normalize input
            qs = qs / torch.norm(qs, p=2) # (N, D)
            ks = ks / torch.norm(ks, p=2) # (N, D)

            # numerator
            q_block=to_block(qs,n_nodes) # (N, B*D)
            k_block_T=to_block(ks,n_nodes).T # (B*D, N)
            v_block=to_block(vs,n_nodes) # (N, B*D)
            kv_block=torch.matmul(k_block_T,v_block) # (B*M, B*D)
            qkv_block=torch.matmul(q_block,kv_block) # (N, B*D)
            qkv=unpack_block(qkv_block,self.out_channels,n_nodes) # (N, D)

            v_sum=v_block.sum(dim=0) # (B*D,)
            v_sum=batch_repeat(v_sum,self.out_channels,n_nodes) # (N, D)
            numerator=qkv+v_sum # (N, D)

            # denominator
            one_list=[]
            for n in n_nodes:
                one=torch.ones((n,1))
                one_list.append(one)
            one_block=torch.block_diag(*one_list).to(qs.device)
            k_sum_block=torch.matmul(k_block_T,one_block) # (B*D, B)
            denom_block=torch.matmul(q_block,k_sum_block) # (N, B)
            denominator=unpack_block(denom_block,1,n_nodes) # (N, 1)
            denominator+=batch_repeat(n_nodes,1,n_nodes) # (N, 1)

            attention=numerator/denominator # (N, D)
        
        elif kernel=='simple2': # equivalent to but faster than simple
            start_row=0
            qs = qs / torch.norm(qs, p=2) # (N, D)
            ks = ks / torch.norm(ks, p=2) # (N, D)
            attn_list=[]
            for n in n_nodes:
                cur_q=qs[start_row:start_row+n]
                cur_k_T=ks[start_row:start_row+n].T
                cur_v=vs[start_row:start_row+n]

                kv=torch.matmul(cur_k_T, cur_v)
                qkv=torch.matmul(cur_q, kv)
                numerator=qkv+torch.sum(cur_v,dim=0,keepdim=True)

                k_sum=cur_k_T.sum(dim=-1,keepdim=True)
                denominator=torch.matmul(cur_q,k_sum)
                denominator+=n

                cur_attn=numerator/denominator
                attn_list.append(cur_attn)

                start_row+=n
            attention=torch.cat(attn_list,dim=0)
        elif kernel=='sigmoid': # equivalent to but faster than simple
            start_row=0
            attn_list=[]

            for n in n_nodes:
                cur_q=qs[start_row:start_row+n]
                cur_k_T=ks[start_row:start_row+n].T
                cur_v=vs[start_row:start_row+n]

                sig_qk=torch.sigmoid(torch.matmul(cur_q,cur_k_T))
                numerator=torch.matmul(sig_qk,cur_v)
                denominator=torch.sum(sig_qk,dim=-1,keepdim=True)

                cur_attn=numerator/denominator
                attn_list.append(cur_attn)

                start_row+=n
            attention=torch.cat(attn_list,dim=0)
        else:
            raise ValueError
        
        return attention

    def forward(self, query_input, source_input, n_nodes, edge_index=None, edge_weight=None):
        '''
        n_nodes: (B,) B is batch_size, it indicates the number of nodes in each graph.
        in_channels should equal to out_channels when use_weight is False
        '''
        query = self.Wq(query_input)
        key = self.Wk(source_input)
        if self.use_weight:
            value = self.Wv(source_input)

        attention_output=self.full_attention(query,key,value,self.kernel,n_nodes)

        if self.use_graph:
            if self.graph_weight>0:
                final_output = (1-self.graph_weight)*attention_output + self.graph_weight*gcn_conv(value, edge_index, edge_weight)
            else:
                final_output = attention_output + gcn_conv(value, edge_index, edge_weight)
        else:
            final_output=attention_output
        
        return final_output
    
class Difformer(nn.Module):
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
    
    def forward(self, data):
        x=data.x
        edge_index=data.edge_index
        n_nodes=data.n_nodes
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

