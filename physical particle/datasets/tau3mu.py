import sys
sys.path.append('../')

import os
import yaml
import shutil
import os.path as osp
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch, pickle
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph, radius_graph

from utils import log, get_random_idx_split, download_url, extract_zip, decide_download


class Tau3Mu(InMemoryDataset):
    def __init__(self, root, data_config, seed):
        self.url_raw = 'https://zenodo.org/record/7265547/files/tau3mu_raw.zip'
        self.url_processed = 'https://zenodo.org/record/7265547/files/tau3mu_processed.zip'
        self.split = data_config['split']
        self.other_features = data_config['other_features']
        self.seed = seed

        self.sample_filters = data_config['sample_filters']
        self.hit_filters = data_config['hit_filters']

        super().__init__(root=root)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[1]
        self.pos_dim = self.data.pos.shape[1]
        self.signal_class = 1
        self.dataset_name = 'tau3mu'
        self.feature_type = data_config['feature_type']

        if self.feature_type == 'only_pos':
            node_scalar_feat = self.pos_dim
        elif self.feature_type == 'only_x':
            node_scalar_feat = self.x_dim
        elif self.feature_type == 'only_ones':
            node_scalar_feat = 1
        else:
            assert self.feature_type == 'both_x_pos'
            node_scalar_feat = self.x_dim + self.pos_dim

        self.feat_info = {'node_categorical_feat': [], 'node_scalar_feat': node_scalar_feat}

    @property
    def raw_file_names(self):
        return ['tau3mu_mixed.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        if osp.exists(self.processed_paths[0]):
            return
        if decide_download(self.url_raw, is_raw=True):
            path = download_url(self.url_raw, self.root)
            extract_zip(path, self.root)
            os.unlink(path)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):

        df = pd.read_pickle(self.raw_dir + '/tau3mu_mixed.pkl')
        # with open(self.raw_dir + '/tau3mu_mixed.pkl','rb') as f:
        #     df=pickle.load(f,encoding='latin1')

        log('[INFO] Processing entries...')
        data_list = []
        for entry in tqdm(df.itertuples(), total=len(df)):
            entry = self.mask_hits(entry, self.hit_filters, self.sample_filters)
            if entry is None:
                continue
            x = torch.tensor(np.stack([entry[feature] for feature in self.other_features], axis=1)).float()
            pos = self.get_pos(entry)
            # y = torch.tensor(entry['y']).float().view(-1, 1)
            y = torch.tensor(entry['y']).view(-1, 1)

            # add pos info to features
            x=torch.cat((x, pos),dim=1)

            if y.item() == 1:
                node_label = torch.tensor(entry['node_label']).float().view(-1)
            else:
                node_label = torch.zeros((x.shape[0])).float()

            edge_index = radius_graph(pos, r=1.0, loop=True)

            data = Data(x=x, pos=pos, y=y, edge_index=edge_index, node_label=node_label, n_nodes=x.shape[0])
            data_list.append(data)

        idx_split = get_random_idx_split(len(data_list), self.split, self.seed)
        data, slices = self.collate(data_list)

        log('[INFO] Saving data.pt...')
        torch.save((data, slices, idx_split), self.processed_paths[0])

    @staticmethod
    def get_pos(entry):
        x = entry['mu_hit_sim_eta'].reshape(-1, 1)
        y = np.deg2rad(entry['mu_hit_sim_phi'].reshape(-1, 1))
        return torch.tensor(np.concatenate((x, y), axis=1)).float()

    @staticmethod
    def mask_hits(entry, hit_filters, sample_filters):
        mask = np.ones(entry.n_mu_hit, dtype=bool)
        for k, v in hit_filters.items():
            assert isinstance(getattr(entry, k), np.ndarray)
            mask *= eval('entry.' + k + v)

        masked_entry = {'n_mu_hit': mask.sum()}
        for k in entry._fields:
            value = getattr(entry, k)
            if isinstance(value, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k:
                assert value.shape[0] == entry.n_mu_hit
                assert 'mu_hit' in k or k == 'node_label'
                masked_entry[k] = value[mask].reshape(-1)
            else:
                if k != 'n_mu_hit':
                    masked_entry[k] = value

        if masked_entry['y'] == 1 and not eval("masked_entry['node_label'].sum()" + sample_filters['num_hits']):
            return None
        if masked_entry['y'] == 0 and not eval("masked_entry['n_mu_hit']" + sample_filters['num_hits']):
            return None
        return masked_entry


if __name__ == '__main__':
    data_config = yaml.safe_load(open('../configs/tau3mu.yml'))['data']
    dataset = Tau3Mu(root='../../data/tau3mu', data_config=data_config, seed=42)
