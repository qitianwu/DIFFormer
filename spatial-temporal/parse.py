from gnns import *
from difformer import DIFFormer


def parse_method(args, n, c, d, device):
    if args.method == 'link':
        model = LINK(n, c).to(device)
        suffix = ''
    elif args.method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'bn' + str(args.use_bn)
    elif args.method == 'mlp' or args.method == 'cs':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers)
    elif args.method == 'sgc':
        if args.cached:
            model = SGC(in_channels=d, out_channels=c, hops=args.hops).to(device)
        else:
            model = SGCMem(in_channels=d, out_channels=c,
                           hops=args.hops).to(device)
        suffix = 'hops' + str(args.hops) + 'cached' + str(args.cached)
    elif args.method == 'gprgnn':
        model = GPRGNN(d, args.hidden_channels, c, alpha=args.gpr_alpha).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + '100alpha' + str(int(100*args.alpha))
    elif args.method == 'appnp':
        model = APPNP_Net(d, args.hidden_channels, c, alpha=args.gpr_alpha).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + '100alpha' + str(int(100*args.alpha))
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'bn' + str(args.use_bn) +\
            'heads' + str(args.gat_heads) + 'out_heads' + str(args.out_heads)
    elif args.method == 'mixhop':
        model = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers,
                       dropout=args.dropout, hops=args.hops).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'hops' + str(args.hops)
    elif args.method == 'gcnjk':
        model = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, jk_type=args.jk_type).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'jk_type' + str(args.jk_type)
    elif args.method == 'gatjk':
        model = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, heads=args.gat_heads,
                        jk_type=args.jk_type).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'heads' + str(args.gat_heads) + 'jk_type' + str(args.jk_type)
    elif args.method == 'difformer':
        model=DIFFormer(d,args.hidden_channels, c, num_layers=args.num_layers, alpha=args.alpha, dropout=args.dropout, num_heads=args.num_heads, kernel=args.kernel,
                       use_bn=args.use_bn, use_residual=args.use_residual, use_graph=args.use_graph, use_weight=args.use_weight).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + '100alpha' + str(int(100*args.alpha)) + 'kernel' + args.kernel
        suffix += 'heads' + str(args.num_heads) + 'bn' + str(args.use_bn) + 'residual' + str(args.use_residual) + 'graph' + str(args.use_graph) + 'weight' + str(args.use_weight)
    elif args.method == 'mpnnlstm':
        model = MPNN_LSTM(d, args.hidden_channels, c, n, 1, args.dropout).to(device)
        suffix = 'hidden' + str(args.hidden_channels)
    elif args.method == 'dcrnn':
        model = DC_RNN(d, args.hidden_channels, c, args.dcrnn_filters).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'filter' + str(args.dcrnn_filters)
    else:
        raise ValueError('Invalid method')
    return model, suffix


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='chickenpox')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', type=int, default=20)
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=-1, help='-1 means same as train_ratio')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')

    # difformer model
    parser.add_argument('--method', type=str, default='difformer')
    parser.add_argument('--special_treat', type=str, default='None')
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--kernel', type=str, default='simple', choices=['simple', 'sigmoid'])

    # baseline model
    parser.add_argument('--gat_heads', type=int, default=2,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--dcrnn_filters', type=int, default=1,
                        help='filter K for DCRNN')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.0)

    # utilities
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--debug','-D', action='store_true',
                        help='debug mode')
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--save_result', action='store_true',
                        help='save result')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='../../model/')


