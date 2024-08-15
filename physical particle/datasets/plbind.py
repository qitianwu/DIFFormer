# Adopted from EquiBind
# https://github.com/HannesStark/EquiBind/blob/main/datasets/pdbbind.py

import sys
sys.path.append('../')

import os
import yaml
import shutil
import pickle
import warnings
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

import math
import numpy as np
import pandas as pd
from scipy import spatial
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph

import pint
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from utils import pmap_multi, disable_rdkit_logging, safe_index, log, allowable_features, sr
from utils import download_url, extract_zip, decide_download


biopython_parser = PDBParser()
ureg = pint.UnitRegistry()
disable_rdkit_logging()


def binary_affinity(affinity, thres=100):
    aff_nM = (affinity * ureg.M).to(ureg.nM).magnitude
    return (aff_nM < thres).float()


class GenContact(BaseTransform):
    def __init__(self, data_config):
        self.bin_thres = data_config['bin_thres']

    def __call__(self, data):
        data.y = binary_affinity(data.affinity, thres=self.bin_thres)
        if data.y.item() == 0:
            return data

        # contact_idx = np.argwhere(data.contact_label == 1).reshape(-1)
        # data.node_label[contact_idx] = 1
        return data


class PLBind(InMemoryDataset):

    def __init__(self, root, data_config, n_jobs=32, debug=False):
        self.url_raw = 'https://zenodo.org/record/7265547/files/plbind_raw.zip'
        self.url_processed = 'https://zenodo.org/record/7265547/files/plbind_processed.zip'
        self.data_dir = root
        self.use_rdkit_coords = data_config['use_rdkit_coords']
        self.pocket_cutoff = data_config['pocket_cutoff']
        self.bin_thres = data_config['bin_thres']
        self.n_jobs = n_jobs
        self.debug = debug

        super().__init__(root, transform=GenContact(data_config))
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.complex_names = pickle.load(open(os.path.join(self.processed_dir, 'raw_data.pkl'), 'rb'))[-1]
        self.signal_class = 1
        self.dataset_name = 'plbind'
        # self.feature_type = data_config['feature_type']

        # self.x_dim = self.data.x.shape[1]
        # self.pos_dim = self.data.pos.shape[1]

        # self.x_lig_dim = self.data.x_lig.shape[1]
        # self.pos_lig_dim = self.data.pos_lig.shape[1]

        node_categorical_feat = [len(allowable_features['possible_amino_acids'])]
        # node_categorical_feat_lig = [len(allowable_features['possible_atomic_num_list']), len(allowable_features['possible_chirality_list']), len(allowable_features['possible_degree_list']),
        #                              len(allowable_features['possible_formal_charge_list']), len(allowable_features['possible_implicit_valence_list']), len(allowable_features['possible_numH_list']),
        #                              len(allowable_features['possible_number_radical_e_list']), len(allowable_features['possible_hybridization_list']), len(allowable_features['possible_is_aromatic_list']),
        #                              len(allowable_features['possible_numring_list']), len(allowable_features['possible_is_in_ring3_list']), len(allowable_features['possible_is_in_ring4_list']),
        #                              len(allowable_features['possible_is_in_ring5_list']), len(allowable_features['possible_is_in_ring6_list']), len(allowable_features['possible_is_in_ring7_list']),
        #                              len(allowable_features['possible_is_in_ring8_list'])]

        # if self.feature_type == 'only_pos':
        #     node_scalar_feat_lig = self.pos_lig_dim
        #     node_scalar_feat = self.pos_dim
        #     node_categorical_feat, node_categorical_feat_lig = [], []
        # elif self.feature_type == 'only_x':
        #     node_scalar_feat_lig = self.x_lig_dim - 16  # 16 categorical features
        #     node_scalar_feat = self.x_dim - 1
        # elif self.feature_type == 'only_ones':
        #     node_scalar_feat_lig = 1
        #     node_scalar_feat = 1
        #     node_categorical_feat, node_categorical_feat_lig = [], []
        # else:
        #     assert self.feature_type == 'both_x_pos'
        #     node_scalar_feat_lig = self.x_lig_dim - 16 + self.pos_lig_dim
        #     node_scalar_feat = self.x_dim - 1 + self.pos_dim

        # self.feat_info = {'node_categorical_feat': node_categorical_feat, 'node_scalar_feat': node_scalar_feat}
        # self.feat_info_lig = {'node_categorical_feat': node_categorical_feat_lig, 'node_scalar_feat': node_scalar_feat_lig}
        # self.n_categorical_feat_to_use_lig = data_config['n_categorical_feat_to_use_lig']
        # self.n_scalar_feat_to_use_lig = data_config['n_scalar_feat_to_use_lig']
        # self.n_categorical_feat_to_use = data_config['n_categorical_feat_to_use']
        # self.n_scalar_feat_to_use = data_config['n_scalar_feat_to_use']

    @property
    def raw_file_names(self):
        return ['contacts', 'index', 'pdb', 'split']

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
        self.contacts_dir = os.path.join(self.data_dir,'raw','contacts')
        self.index_path = os.path.join(self.data_dir , 'raw' , 'index' , 'INDEX_general_PL_data.2020')
        col_names = ['PDB code', 'resolution', 'release year', '-logKd/Ki', 'Kd/Ki', 'ligand name']
        self.index = pd.read_csv(self.index_path, sep='\s\s+', skiprows=list(range(6)), header=None, names=col_names, engine='python')

        complex_names = get_clean_complex_names(Path(self.raw_dir))
        useable_complex_names = deepcopy(complex_names)
        if not os.path.exists(os.path.join(self.processed_dir, 'raw_data.pkl')):
            ligs, rec_paths, affinities = [], [], []
            for name in tqdm(complex_names, desc='Reading raw data'):
                affinity = self.unit_check(self.index[self.index['PDB code'] == name]['Kd/Ki'].item())
                if affinity is False:
                    useable_complex_names.remove(name)
                    continue

                lig_path = os.path.join(self.raw_dir , 'pdb' , name)
                lig = read_molecule(str(os.path.join(lig_path , f'{name}_ligand.sdf')), sanitize=True, remove_hs=False)
                if lig == None:  # read mol2 file if sdf file cannot be sanitized
                    lig = read_molecule(str(os.path.join(lig_path , f'{name}_ligand.mol2')), sanitize=True, remove_hs=False)

                ligs.append(lig)
                affinities.append(torch.tensor(affinity, dtype=torch.float).view(1, 1))
                rec_paths.append(os.path.join(lig_path , f'{name}_protein_processed.pdb'))
            complex_names = useable_complex_names
            log('Saving raw data...')
            pickle.dump((ligs, rec_paths, affinities, complex_names), open(os.path.join(self.processed_dir, 'raw_data.pkl'), 'wb'))
        else:
            log('Loading raw data...')
            ligs, rec_paths, affinities, complex_names = pickle.load(open(os.path.join(self.processed_dir, 'raw_data.pkl'), 'rb'))
        log(f'{len(complex_names)} complexes are usable.')

        if self.debug:
            rec_paths, ligs, complex_names = rec_paths[:100], ligs[:100], complex_names[:100]

        if not os.path.exists(os.path.join(self.processed_dir, 'recs_coors.pkl')):
            receptor_representatives = pmap_multi(get_receptor, zip(rec_paths, ligs), n_jobs=1, cutoff=10, desc='Get receptors')
            recs, recs_coords, c_alpha_coords, n_coords, c_coords, res_nos, chain_ids = map(list, zip(*receptor_representatives))
            log('Saving receptor coordinates...')
            pickle.dump((recs, recs_coords, c_alpha_coords, n_coords, c_coords, res_nos, chain_ids), open(os.path.join(self.processed_dir, 'recs_coors.pkl'), 'wb'))
        else:
            log('Loading receptor coordinates...')
            recs, recs_coords, c_alpha_coords, n_coords, c_coords, res_nos, chain_ids = pickle.load(open(os.path.join(self.processed_dir, 'recs_coors.pkl'), 'rb'))

        if not os.path.exists(os.path.join(self.processed_dir, 'rec_graphs.pkl')):
            rec_graphs = pmap_multi(get_calpha_data, zip(recs, c_alpha_coords), n_jobs=self.n_jobs, desc='Convert receptors to graphs')
            log('Saving receptor graphs...')
            pickle.dump(rec_graphs, open(os.path.join(self.processed_dir, 'rec_graphs.pkl'), 'wb'))
        else:
            log('Loading receptor graphs...')
            rec_graphs = pickle.load(open(os.path.join(self.processed_dir, 'rec_graphs.pkl'), 'rb'))

        # if not os.path.exists(os.path.join(self.processed_dir, 'lig_graphs.pkl')):
        #     lig_graphs = pmap_multi(get_lig_data, zip(ligs, complex_names), n_jobs=1, use_rdkit_coords=self.use_rdkit_coords, debug=self.debug, desc='Convert ligands to graphs')
        #     log('Saving ligand graphs...')
        #     pickle.dump(lig_graphs, open(os.path.join(self.processed_dir, 'lig_graphs.pkl'), 'wb'))
        # else:
        #     log('Loading ligand graphs...')
        #     lig_graphs = pickle.load(open(os.path.join(self.processed_dir, 'lig_graphs.pkl'), 'rb'))

        # if not os.path.exists(os.path.join(self.processed_dir, 'pockets.pkl')):
        #     pockets = pmap_multi(get_pocket_nodes, zip(lig_graphs, rec_graphs, res_nos, chain_ids, complex_names), n_jobs=1, cutoff=self.pocket_cutoff, contacts_dir=self.contacts_dir, desc='Get pockets')
        #     pocket_nodes, pocket_contacts = map(list, zip(*pockets))
        #     log('Saving pocket nodes...')
        #     pickle.dump((pocket_nodes, pocket_contacts), open(os.path.join(self.processed_dir, 'pockets.pkl'), 'wb'))
        # else:
        #     log('Loading pocket nodes...')
        #     pocket_nodes, pocket_contacts = pickle.load(open(os.path.join(self.processed_dir, 'pockets.pkl'), 'rb'))

        data_list = []
        for idx, name in tqdm(enumerate(complex_names), desc='Processing data'):
            # lig_graph = lig_graphs[idx]
            rec_graph = rec_graphs[idx]
            affinity = affinities[idx]

            # node_label = pocket_nodes[idx]
            # contact_label = pocket_contacts[idx]

            rec_graph.pos = (rec_graph.pos - rec_graph.pos.mean(dim=0, keepdim=True))  # center the graph
            # lig_graph.pos = (lig_graph.pos - lig_graph.pos.mean(dim=0, keepdim=True))  # center the graph

            y = binary_affinity(affinity, thres=self.bin_thres)

            # add pos to x
            x=torch.cat((rec_graph.x,rec_graph.pos),dim=1)
            n_nodes=x.shape[0]

            edge_index = knn_graph(rec_graph.pos, k=5, flow='target_to_source', loop=True)

            data = Data(x=rec_graph.x, edge_index=edge_index, pos=rec_graph.pos, true_pos=rec_graph.true_pos,
                        affinity=affinity, y=y, n_nodes=n_nodes)
            data_list.append(data)

            


        idx_split = self.get_idx_split(self.data_dir, complex_names)
        data, slices = self.collate(data_list)
        torch.save((data, slices, idx_split), self.processed_paths[0])

    def unit_check(self, affinity):
        if 'IC' in affinity:
            return False
        elif '>' in affinity or '<' in affinity:
            return False
        elif '~' in affinity:
            aff_value = affinity.split('~')[-1]
        elif '=' in affinity:
            aff_value = affinity.split('=')[-1]
        else:
            raise ValueError(f'Affinity {affinity} is not in the correct format.')

        unit = aff_value.split('//')[0][-2:]
        aff_value = float(aff_value.split('//')[0][:-2])
        return (aff_value * ureg[unit]).to(ureg['M']).magnitude

    @staticmethod
    def get_idx_split(data_dir, complex_names):
        train_names = open(os.path.join(data_dir , 'raw' , 'split' , 'timesplit_no_lig_overlap_train')).read().splitlines()
        valid_names = open(os.path.join(data_dir , 'raw' , 'split' , 'timesplit_no_lig_overlap_val')).read().splitlines()
        test_names = open(os.path.join(data_dir , 'raw' , 'split' , 'timesplit_test')).read().splitlines()

        train_idx, valid_idx, test_idx, unused_idx = [], [], [], []
        for name in complex_names:
            if name in train_names:
                train_idx.append(complex_names.index(name))
            elif name in valid_names:
                valid_idx.append(complex_names.index(name))
            elif name in test_names:
                test_idx.append(complex_names.index(name))
            else:
                unused_idx.append(complex_names.index(name))
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx, 'unused': unused_idx}


def get_clean_complex_names(raw_dir):
    complex_names = os.listdir(os.path.join(raw_dir , 'pdb'))
    complex_names = [each for each in complex_names if os.listdir(os.path.join(raw_dir , 'pdb', each)) != []]
    assert '1a50' not in complex_names  # no files
    for each in ['3m1s', '3q4c']:  # cannot pickle
        if each in complex_names:
            complex_names.remove(each)
    return complex_names


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None
    return mol


def get_pocket_nodes(lig_graph, rec_graph, res_no, chain_id, complex_name, contacts_dir, cutoff):
    node_label = torch.zeros(rec_graph.num_nodes).float()
    lig_rec_distance = spatial.distance.cdist(lig_graph.true_pos, rec_graph.true_pos)
    signal_idx = np.argwhere(lig_rec_distance.min(0) < cutoff).reshape(-1)
    node_label[signal_idx] = 1

    contact_label = torch.zeros(rec_graph.num_nodes).float()
    res_info = np.array(list(zip(res_no, chain_id)))
    contacts = pickle.load(open(os.path.join(contacts_dir ,complex_name + '.pkl'), 'rb'))
    df = pd.DataFrame([], columns=['res_no_rec', 'chain_rec'])
    for each in contacts:
        if each['hydrogen_df'] is not None:
            df = pd.concat([df, each['hydrogen_df'][['res_no_rec', 'chain_rec']].drop_duplicates()], ignore_index=True)
        if each['non_bond_df'] is not None:
            df = pd.concat([df, each['non_bond_df'][['res_no_rec', 'chain_rec']].drop_duplicates()], ignore_index=True)
    df['res_no_rec'] = [int(each) if str(each).lstrip('-').isdigit() else int(each[:-1]) for each in df['res_no_rec']]
    contact_res = df.drop_duplicates().values
    contact_idx = []
    for each_contact in contact_res:
        res = np.where((res_info[:, 0] == str(each_contact[0])) & (res_info[:, 1] == each_contact[1]))[0]
        contact_idx.extend(list(res))
    contact_label[contact_idx] = 1
    return node_label, contact_label


def get_receptor(rec_path, lig, cutoff):
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]  # use the first model

    min_distances, coords, c_alpha_coords, res_nos, chain_ids, n_coords, c_coords, valid_chain_ids, lengths = ([] for i in range(9))
    for i, chain in enumerate(rec):
        chain_coords, chain_c_alpha_coords, chain_n_coords, chain_c_coords, chain_res_nos, chain_chain_ids, chain_is_water, invalid_res_ids = ([] for i in range(8))
        chain_is_water = False
        count = 0
        for _, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                chain_is_water = True
            residue_coords = []
            c_alpha, n, c = None, None, None
            res_no = residue.get_id()[1]
            assert isinstance(res_no, int)
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))
            # TODO: Also include the chain_coords.append(np.array(residue_coords)) for non amino acids such that they can be used when using the atom representation of the receptor
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not some weired molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha), chain_n_coords.append(n), chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords)), chain_res_nos.append(res_no), chain_chain_ids.append(chain.get_id())
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min()
        else:
            min_distance = np.inf
        if chain_is_water:
            min_distances.append(np.inf)
        else:
            min_distances.append(min_distance)
        lengths.append(count), coords.append(chain_coords), c_alpha_coords.append(np.array(chain_c_alpha_coords))
        res_nos.append(np.array(chain_res_nos)), chain_ids.append(np.array(chain_chain_ids))
        n_coords.append(np.array(chain_n_coords)), c_coords.append(np.array(chain_c_coords))
        if min_distance < cutoff and not chain_is_water:
            valid_chain_ids.append(chain.get_id())
    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(np.argmin(min_distances))

    valid_coords, valid_c_alpha_coords, valid_n_coords, valid_c_coords, valid_res_nos, valid_chain_ids_each_res, valid_lengths, invalid_chain_ids = ([] for i in range(8))
    for i, chain in enumerate(rec):  # only use one chain that is closest to the ligand
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i]), valid_c_alpha_coords.append(c_alpha_coords[i])
            valid_res_nos.append(res_nos[i]), valid_chain_ids_each_res.append(chain_ids[i])
            valid_n_coords.append(n_coords[i]), valid_c_coords.append(c_coords[i]), valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    res_nos = np.concatenate(valid_res_nos, axis=0)  # [n_residues]
    chain_ids = np.concatenate(valid_chain_ids_each_res, axis=0)  # [n_residues]

    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(res_nos)
    assert len(c_alpha_coords) == len(chain_ids)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    return rec, coords, c_alpha_coords, n_coords, c_coords, res_nos, chain_ids


def get_calpha_data(rec, c_alpha_coords):
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    node_attr = rec_residue_featurizer(rec)

    residue_representatives_loc_list = []
    for i, residue in enumerate(rec.get_residues()):
        c_alpha_coord = c_alpha_coords[i]
        residue_representatives_loc_list.append(c_alpha_coord)
    residue_representatives_loc_feat = np.stack(residue_representatives_loc_list, axis=0)  # (N_res, 3)
    pos = torch.from_numpy(residue_representatives_loc_feat.astype(np.float32))
    return Data(x=node_attr, pos=pos, true_pos=pos)


def get_lig_data(mol, name, use_rdkit_coords, debug):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    if use_rdkit_coords:
        try:
            rdkit_coords = get_rdkit_coords(mol, debug).numpy()
            R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
            lig_coords = ((R @ (rdkit_coords).T).T + t.squeeze())
            if debug:
                log('kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
        except Exception as e:
            lig_coords = true_lig_coords
            with open('temp_create_dataset_rdkit_timesplit_no_lig_or_rec_overlap_train.log', 'a') as f:
                f.write('Generating RDKit conformer failed for  \n')
                f.write(name)
                f.write('\n')
                f.write(str(e))
                f.write('\n')
                f.flush()
            log('Generating RDKit conformer failed for  ')
            log(name)
            log(str(e))
    else:
        lig_coords = true_lig_coords

    node_attr = lig_atom_featurizer(mol)
    true_pos = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    new_pos = torch.from_numpy(np.array(lig_coords).astype(np.float32)) if use_rdkit_coords else true_pos
    return Data(x=node_attr, pos=new_pos, true_pos=true_pos)


def rec_residue_featurizer(rec):
    feature_list = []
    sr.compute(rec, level="R")
    for residue in rec.get_residues():
        sasa = residue.sasa
        for atom in residue:
            if atom.name == 'CA':
                bfactor = atom.bfactor
        assert not np.isinf(bfactor)
        assert not np.isnan(bfactor)
        assert not np.isinf(sasa)
        assert not np.isnan(sasa)
        feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname()),
                             sasa,
                             bfactor])
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)


def get_rdkit_coords(mol, debug):
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        if debug:
            log('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    conf = mol.GetConformer()
    lig_coords = conf.GetPositions()
    return torch.tensor(lig_coords, dtype=torch.float32)


# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.
def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t


def lig_atom_featurizer(mol):
    ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])

    return torch.tensor(atom_features_list)


if __name__ == '__main__':
    data_dir = '../../../data/cloud_point/plbind'
    data_config = yaml.safe_load(open('../configs/plbind.yml'))
    # data_config = yaml.safe_load(open('../configs/plbind.yml'))['data']
    dataset = PLBind(data_dir, data_config, n_jobs=32, debug=False)
