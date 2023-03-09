import torch

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmin = result[:, 1].argmin().item()
            print(f'Run {run + 1:02d}:')
            print(f'Lowest Train: {result[:, 0].min():.4f}')
            print(f'Lowest Valid: {result[:, 1].min():.4f}')
            print(f'Lowest Test: {result[:, 2].min():.4f}')
            print(f'Chosen epoch: {argmin+1}')
            print(f'Final Train: {result[argmin, 0]:.4f}')
            print(f'Final Test: {result[argmin, 2]:.4f}')
            self.test=result[argmin, 2]
        else:
            result = torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].min().item()
                test1 = r[:, 2].min().item()
                valid = r[:, 1].min().item()

                train2 = r[r[:, 1].argmin(), 0].item()
                test2 = r[r[:, 1].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Lowest Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'Lowest Test: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 2]
            print(f'Lowest Valid: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.4f} ± {r.std():.4f}')

            self.test=r.mean()
            return best_result[:, 4]
    
    def output(self,out_path,info):
        with open(out_path,'a') as f:
            f.write(info)
            f.write(f'test mse:{self.test}\n')

import os
def save_result(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')
    filename = f'results/{args.dataset}/{args.method}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} " + f"{args.kernel}: " + f"{args.use_graph}: " + \
            f"{args.weight_decay} " + f"{args.dropout} " + f"{args.lr} " + \
            f"{results.mean():.4f} $\pm$ {results.std():.4f} \n")