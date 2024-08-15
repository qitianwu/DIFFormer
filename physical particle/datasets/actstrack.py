import sys
sys.path.append('../')

import os
import yaml
import shutil
import pickle
import os.path as osp
from tqdm import tqdm
from itertools import combinations

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph

from utils import get_random_idx_split, download_url, extract_zip, decide_download


class ActsTrack(InMemoryDataset):

    def __init__(self, root, tesla, data_config, seed):
        self.url_raw = 'https://zenodo.org/record/7265547/files/actstrack_raw_2T.zip'
        self.url_processed = 'https://zenodo.org/record/7265547/files/actstrack_processed_2T.zip'
        self.tesla = tesla
        self.split = data_config['split']
        self.sample_tracks = data_config['sample_tracks']
        self.pos_features = data_config['pos_features']
        self.other_features = data_config['other_features']
        self.seed = seed

        self.im_thres = data_config['im_thres']  # invariant mass threshold

        super().__init__(root)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[1]
        self.pos_dim = self.data.pos.shape[1]
        self.feature_type = data_config['feature_type']
        self.signal_class = 1
        self.dataset_name = f'actstrack_{self.tesla}'

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
    def raw_dir(self) -> str:
        return osp.join(self.root, f'raw_{self.tesla}')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_{self.tesla}')

    @property
    def raw_file_names(self):
        return [f'bkg_events_{self.tesla}.pkl', f'signal_events_{self.tesla}.pkl']

    @property
    def processed_file_names(self):
        return [f'data_{self.tesla}.pt']

    def download(self):
        if self.tesla != '2T':
            raise NotImplementedError('Please download datasets with other magnetic field strength at https://zenodo.org/record/7265547')

        if osp.exists(self.processed_paths[0]):
            return
        if decide_download(self.url_raw, is_raw=True):
            path = download_url(self.url_raw, self.root)
            extract_zip(path, self.root)
            os.unlink(path)
        # else:
        #     if decide_download(self.url_processed, is_raw=False):
        #         path = download_url(self.url_processed, self.root)
        #         extract_zip(path, self.root)
        #         os.unlink(path)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        # if decide_download(self.url_processed, is_raw=False):
        #     path = download_url(self.url_processed, self.root)
        #     extract_zip(path, self.root)
        #     os.unlink(path)
        #     return

        signal_events = pickle.load(open(self.raw_dir + f'/signal_events_{self.tesla}.pkl', 'rb'))
        signal_data_list = self.build_data(signal_events, 'signal')

        bkg_events = pickle.load(open(self.raw_dir + f'/bkg_events_{self.tesla}.pkl', 'rb'))
        bkg_data_list = self.build_data(bkg_events, 'bkg')

        data_list = signal_data_list + bkg_data_list
        idx_split = get_random_idx_split(len(data_list), self.split, self.seed)

        data, slices = self.collate(data_list)
        torch.save((data, slices, idx_split), self.processed_paths[0])

    def build_data(self, events, event_type):
        data_list = []
        cnt = 0
        pbar = tqdm(events)
        for initial, _, hits in pbar:
            muons = initial[(initial['particle_type'] == 13) | (initial['particle_type'] == -13)]
            electrons = initial[(initial['particle_type'] == 11) | (initial['particle_type'] == -11)]
            if len(hits) == 0 or len(initial) == 0:
                continue

            hits['node_label'] = 0
            y = torch.tensor(0).float().view(-1, 1)
            signal_im = -1
            signal_particles = []
            if event_type == 'signal':
                if len(muons) < 2 and len(electrons) < 2:
                    continue

                signal_electrons = self.get_signal_particles(electrons, self.im_thres)
                signal_muons = self.get_signal_particles(muons, self.im_thres)
                signal_info = np.array(signal_electrons + signal_muons)
                if signal_info.shape[0] != 1:
                    continue

                signal_particles = list(signal_info[:, :2].reshape(-1))
                signal_im = signal_info[:, 2].item()
                assert len(signal_particles) == 2

                hits.loc[hits['particle_id'].isin(signal_particles), 'node_label'] = 1
                y = torch.tensor(1).float().view(-1, 1)

                if hits['node_label'].sum() == 0:  # no signal hits in tracks, even though there are in the initial position
                    continue

            # sampling tracks
            if self.sample_tracks:
                n_ptcl_to_sample = self.sample_tracks - len(signal_particles)
                to_sample = np.random.choice(hits['particle_id'].unique(), n_ptcl_to_sample)
                ptcl_to_use = list(to_sample) + list(signal_particles)
                hits = hits[hits['particle_id'].isin(ptcl_to_use)].reset_index(drop=True)

            hits['node_id'] = range(len(hits))
            pos = torch.tensor(hits[self.pos_features].to_numpy()).float()
            x = torch.tensor(hits[self.other_features].to_numpy()).float()
            node_label = torch.tensor(hits['node_label'].to_numpy()).float().view(-1)
            node_dir = torch.tensor(hits[['tpx', 'tpy', 'tpz']].to_numpy()).float()
            track_ids = torch.full((len(hits),), -1)  # indices which track the node belongs to

            num_tracks = 0
            all_ptcls = hits['particle_id'].unique()
            for ptcl in all_ptcls:
                track = hits[hits['particle_id'] == ptcl]

                if ptcl in signal_particles:
                    assert np.all(track['node_label'] == 1)
                else:
                    assert np.all(track['node_label'] == 0)
                track_ids[track['node_id'].to_numpy()] = num_tracks
                num_tracks += 1
            assert -1 not in track_ids

            # add pos to x
            x=torch.cat((x,pos),dim=1)
            n_nodes=x.shape[0]

            pos = pos / 2955.5000 * 100
            norm = pos.norm(dim = -1, keepdim = True)
            pos = pos / norm.clamp(min=1e-6)
            
            edge_index = knn_graph(pos, k=5, loop=True)

            data_list.append(Data(x=x, pos=pos, y=y, edge_index=edge_index, node_label=node_label,
                                  node_dir=node_dir, num_tracks=num_tracks, track_ids=track_ids, signal_im=signal_im, n_nodes=n_nodes))
            
            cnt += 1
            pbar.set_description(f'[INFO] Processed {cnt} events')

        return data_list

    @staticmethod
    def invariant_mass(m, px1, py1, pz1, px2, py2, pz2):
        first_term = m**2
        second_term = np.sqrt(m**2 + px1**2 + py1**2 + pz1**2) * np.sqrt(m**2 + px2**2 + py2**2 + pz2**2)
        third_term = px1*px2 + py1*py2 + pz1*pz2
        return np.sqrt(2 * (first_term + second_term - third_term))

    @staticmethod
    def get_signal_particles(particles, thres):
        if len(particles) < 2:
            return []

        res = []
        all_particle_pairs = combinations(range(len(particles)), 2)
        for i, j in all_particle_pairs:
            first_particle, second_particle = particles.iloc[i], particles.iloc[j]
            if first_particle['q'] * second_particle['q'] > 0:
                continue

            # if np.linalg.norm(first_particle[['vx', 'vy', 'vz']] - second_particle[['vx', 'vy', 'vz']], ord=2) < 0.1:
            im = ActsTrack.invariant_mass(first_particle['m'], first_particle['px'], first_particle['py'], first_particle['pz'], second_particle['px'], second_particle['py'], second_particle['pz'])
            if abs(im - 91.1876) < thres:
                res.append([first_particle['particle_id'], second_particle['particle_id'], im])
        return res


if __name__ == '__main__':
    data_config = yaml.safe_load(open('../configs/actstrack.yml'))['data']
    # for tesla in tqdm(['2T', '4T', '6T', '8T', '10T', '12T', '14T', '16T', '18T', '20T']):
    tesla = '2T'
    dataset = ActsTrack(root='../../data/actstrack', tesla=tesla, data_config=data_config, seed=42)
