import sys
sys.path.append('../')

import os
import yaml
import shutil
import os.path as osp
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import knn_graph, radius_graph

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

from utils import download_url, extract_zip, decide_download


class SynMol(InMemoryDataset):
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']

    def __init__(self, root, data_config, seed):
        self.url_raw = 'https://zenodo.org/record/7265547/files/synmol_raw.zip'
        self.url_processed = 'https://zenodo.org/record/7265547/files/synmol_processed.zip'
        self.seed = seed

        super().__init__(root)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[1]
        self.pos_dim = self.data.pos.shape[1]
        self.feature_type = data_config['feature_type']
        self.signal_class = 1
        self.dataset_name = 'synmol'

        node_categorical_feat = [len(self.ATOM_TYPES)]
        if self.feature_type == 'only_pos':
            node_scalar_feat = self.pos_dim
            node_categorical_feat = []
        elif self.feature_type == 'only_x':
            node_scalar_feat = self.x_dim - 1
        elif self.feature_type == 'only_ones':
            node_scalar_feat = 1
            node_categorical_feat = []
        else:
            assert self.feature_type == 'both_x_pos'
            node_scalar_feat = self.x_dim - 1 + self.pos_dim

        self.feat_info = {'node_categorical_feat': node_categorical_feat, 'node_scalar_feat': node_scalar_feat}

    @property
    def raw_file_names(self):
        return [f'logic8_traintest_indices.npz', f'logic8_smiles.csv', 'true_raw_attribution_datadicts.npz', 'x_true.npz', 'y_true.npz']

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

        all_y = np.load(self.raw_dir + '/y_true.npz', allow_pickle=True)['y']
        all_x = np.load(self.raw_dir + '/x_true.npz', allow_pickle=True)['datadict_list'][0]
        all_exp_labels = np.load(self.raw_dir + '/true_raw_attribution_datadicts.npz', allow_pickle=True)['datadict_list']
        mol_df = pd.read_csv(self.raw_dir + '/logic8_smiles.csv')

        raw_idx_split = dict(np.load(self.raw_dir + '/logic8_traintest_indices.npz', allow_pickle=True))
        assert len(all_y) == len(all_x) == len(all_exp_labels) == (raw_idx_split['train_index'].shape[0] + raw_idx_split['test_index'].shape[0])
        split_dict = self.get_split_dict(raw_idx_split, seed=self.seed)

        data_list = []
        idx_split = {'train': [], 'valid': [], 'test': []}
        cnt = 0
        for idx, data in tqdm(enumerate(all_x), total=len(all_x)):
            x = torch.tensor(data['nodes'])
            x = torch.where(x == 1)[1].reshape(-1, 1)
            y = torch.tensor(all_y[idx]).reshape(-1, 1)

            node_label = torch.tensor(all_exp_labels[idx][0]['nodes'][:, -1]).view(-1)
            if all_exp_labels[idx][0]['nodes'].shape[1] > 1:
                assert np.all((all_exp_labels[idx][0]['nodes'][:, :-1].sum(axis=1) > 0) == (all_exp_labels[idx][0]['nodes'][:, -1] == 1))

            mol = Chem.MolFromSmiles(mol_df.iloc[idx]['smiles'])
            m = Chem.AddHs(mol)
            message_id = AllChem.EmbedMolecule(m, randomSeed=self.seed)
            if message_id < 0:
                print(f'Failed to embed molecule {idx}')
                continue
            message_id = AllChem.MMFFOptimizeMolecule(m, maxIters=1000)
            if message_id < 0:
                print(f'Failed to optimize molecule {idx}')
                continue
            m = Chem.RemoveHs(m)
            pos = torch.tensor(m.GetConformer().GetPositions(), dtype=torch.float)
            assert x.shape[0] == m.GetNumAtoms()
            for j in range(m.GetNumAtoms()):
                assert self.ATOM_TYPES[x[j]] == m.GetAtomWithIdx(j).GetSymbol() or m.GetAtomWithIdx(j).GetSymbol() not in self.ATOM_TYPES

            # add pos to x
            x=torch.cat((x,pos),dim=1)
            n_nodes=x.shape[0]
            
            pos = pos * 5.0
            edge_index = knn_graph(pos, k=5, loop=True)

            data_list.append(Data(x=x, pos=pos, y=y, edge_index=edge_index, node_label=node_label, mol_df_idx=idx, n_nodes=n_nodes))
            idx_split[split_dict[idx]].append(cnt)
            cnt += 1

        data, slices = self.collate(data_list)
        torch.save((data, slices, idx_split), self.processed_paths[0])

    @staticmethod
    def get_split_dict(raw_idx_split, seed):
        np.random.seed(seed)

        train_val_idx = raw_idx_split['train_index']
        idx = np.arange(len(train_val_idx))
        np.random.shuffle(idx)

        train_idx = train_val_idx[idx[:-1000]]
        valid_idx = train_val_idx[idx[-1000:]]
        test_idx = raw_idx_split['test_index']

        split_dict = {}
        for idx in train_idx:
            split_dict[idx] = 'train'
        for idx in valid_idx:
            split_dict[idx] = 'valid'
        for idx in test_idx:
            split_dict[idx] = 'test'
        return split_dict


if __name__ == '__main__':
    data_config = yaml.safe_load(open('../configs/synmol.yml'))['data']
    dataset = SynMol(root='../../data/synmol', data_config=data_config, seed=42)
