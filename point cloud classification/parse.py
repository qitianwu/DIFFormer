
from models import *
from difformer import Difformer

def parse_method(args,c,d,device):
    if args.method=='gcn':
        model=GCN(in_channels=d,
                  hidden_channels=args.hidden_channels,
                  out_channels=args.hidden_channels,
                  num_layers=args.num_layers,
                  dropout=args.dropout,
                  use_bn=args.use_bn).to(device)
    elif args.method=='mlp':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=args.hidden_channels, num_layers=args.num_layers,
                    dropout=args.dropout, use_bn=args.use_bn).to(device)
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, args.hidden_channels, num_layers=args.num_layers,
                    dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads).to(device)
    elif args.method=='difformer':
        model=Difformer(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=args.hidden_channels,
                        num_layers=args.num_layers,
                        kernel=args.kernel,
                        alpha=args.alpha,
                        dropout=args.dropout,
                        use_bn=args.use_bn,
                        use_residual=args.use_residual,
                        use_weight=args.use_weight,
                        use_graph=args.use_graph,
                        graph_weight=args.graph_weight).to(device)
    else:
        raise ValueError
    
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../../../NodeFormer/data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=200, help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=500, help='Total number of test')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')

    # model
    parser.add_argument('--method', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    # parser.add_argument('--kernel', type=str, default='simple', choices=['simple', 'sigmoid'])
    parser.add_argument('--kernel', type=str, default='simple')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--graph_weight', type=float, default=-1,
                        help='Graph weight. -1 means add transformer part and GNN part directly.')
    parser.add_argument('--graph_pooling', type=str, default='mean', choices=['mean','max','sum'])

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.0)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')

    parser.add_argument('--k', type=int, default=5, help='KNN parameterss.')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model.')

    #glcn
    parser.add_argument('--graph_learn_hidden', type=int, default=70, help='graph learn hidden')
    parser.add_argument('--degree_control', type=float, default=0.01, help='degree control')
    parser.add_argument('--structure_control', type=float, default=0.01, help='close to structure')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='regularization weight')

    # manireg
    parser.add_argument('--manireg', type=float, default=1.0, help='weight for regularization in ManiReg')
    # lp
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')