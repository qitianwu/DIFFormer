import torch
from collections import defaultdict

printable_method={'transgnn','gat'}

def create_print_dict(args):
    if args.method=='transgnn':
        return {'n_layer':args.num_layers,
        'hidden_channels':args.hidden_channels,
        'trans_heads':args.trans_heads,
        'lr':args.lr,
        'epochs':args.epochs}
    elif args.method=='gat':
        return {'n_layer':args.num_layers,
        'hidden_channels':args.hidden_channels,
        'gat_heads':args.gat_heads,
        'lr':args.lr,
        'epochs':args.epochs
        }
    else:
        return None

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin

            print_str=f'Run {run + 1:02d}:'+\
                f'Highest Train: {result[:, 0].max():.2f} '+\
                f'Highest Valid: {result[:, 1].max():.2f} '+\
                f'Highest Test: {result[:, 2].max():.2f} '+\
                f'Chosen epoch: {ind+1}\n'+\
                f'Final Train: {result[ind, 0]:.2f} '+\
                f'Final Test: {result[ind, 2]:.2f}'
            print(print_str)

        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test = r.mean()
            return best_result[:, 4]

    def output(self, out_path, info):
        with open(out_path, 'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')

import os
def save_result(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')
    filename = f'results/{args.dataset}/{args.method}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} " + f"{args.kernel}: " + f"{args.weight_decay} " + f"{args.dropout} " + \
            f"{args.num_layers} " + f"{args.alpha}: " + f"{args.hidden_channels}: " + \
            f"{results.mean():.2f} $\pm$ {results.std():.2f} \n")

