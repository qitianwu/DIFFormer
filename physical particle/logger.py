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
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            ind = argmax

            print_str=f'Run {run + 1:02d}:'+\
                f'Highest Train: {result[:, 0].max():.2f} '+\
                f'Highest Valid: {result[:, 1].max():.2f} '+\
                f'Highest Test: {result[:, 2].max():.2f} '+\
                f'Chosen epoch: {ind+1}\n'+\
                f'Final Train: {result[ind, 0]:.2f} '+\
                f'Final Test: {result[ind, 2]:.2f}'
            print(print_str)
            
            # print(f'Run {run + 1:02d}:')
            # print(f'Highest Train: {result[:, 0].max():.2f}')
            # print(f'Highest Valid: {result[:, 1].max():.2f}')
            # print(f'Highest Test: {result[:, 2].max():.2f}')
            # print(f'Chosen epoch: {ind+1}')
            # print(f'Final Train: {result[ind, 0]:.2f}')
            # print(f'Final Test: {result[ind, 2]:.2f}')
            # self.test=result[ind, 2]
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            max_val_epoch=0
            for r in result:
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test2 = r[r[:, 1].argmax(), 2].item()
                max_val_epoch=r[:, 1].argmax()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print_str=f'All runs: '
            r = best_result[:, 0]
            print_str+=f'Highest Train: {r.mean():.2f} ± {r.std():.2f} '
            print_str+=f'Highest val epoch:{max_val_epoch}\n'
            r = best_result[:, 1]
            print_str+=f'Highest Test: {r.mean():.2f} ± {r.std():.2f} '
            r = best_result[:, 4]
            print_str+=f'Final Test: {r.mean():.2f} ± {r.std():.2f}'

            # print(f'All runs:')
            # r = best_result[:, 0]
            # print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            # r = best_result[:, 1]
            # print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            # r = best_result[:, 2]
            # print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            # r = best_result[:, 3]
            # print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            # r = best_result[:, 4]
            # print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test=r.mean()
        return print_str
    
    def output(self,out_path,info):
        with open(out_path,'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')



class SimpleLogger(object):
    """ Adapted from https://github.com/CUAI/CorrectAndSmooth """
    def __init__(self, desc, param_names, num_values=2):
        self.results = defaultdict(dict)
        self.param_names = tuple(param_names)
        self.used_args = list()
        self.desc = desc
        self.num_values = num_values
    
    def add_result(self, run, args, values): 
        """Takes run=int, args=tuple, value=tuple(float)"""
        assert(len(args) == len(self.param_names))
        assert(len(values) == self.num_values)
        self.results[run][args] = values
        if args not in self.used_args:
            self.used_args.append(args)
    
    def get_best(self, top_k=1):
        all_results = []
        for args in self.used_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)[-1]
            results_std = results.std(dim=0)

            all_results.append((args, results_mean))
        results = sorted(all_results, key=lambda x: x[1], reverse=True)[:top_k]
        return [i[0] for i in results]
            
    def prettyprint(self, x):
        if isinstance(x, float):
            return '%.2f' % x
        return str(x)
        
    def display(self, args = None):
        
        disp_args = self.used_args if args is None else args
        if len(disp_args) > 1:
            print(f'{self.desc} {self.param_names}, {len(self.results.keys())} runs')
        for args in disp_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)
            results_std = results.std(dim=0)
            res_str = f'{results_mean[0]:.2f} ± {results_std[0]:.2f}'
            for i in range(1, self.num_values):
                res_str += f' -> {results_mean[i]:.2f} ± {results_std[1]:.2f}'
            print(f'Args {[self.prettyprint(x) for x in args]}: {res_str}')
        if len(disp_args) > 1:
            print()
        return results