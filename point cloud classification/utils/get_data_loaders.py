import os
from pathlib import Path
from torch_geometric.loader import DataLoader
from datasets import ActsTrack, PLBind, Tau3Mu, SynMol


def get_data_loaders(dataset_name, batch_size, data_config, seed):
    data_dir = '../../data/cloud_point'
    assert dataset_name in ['tau3mu', 'plbind', 'synmol'] or 'acts' in dataset_name

    if 'actstrack' in dataset_name:
        dataset_dir, tesla = dataset_name, '2T'
        dataset = ActsTrack(os.path.join(data_dir,dataset_name), tesla=tesla, data_config=data_config, seed=seed)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, idx_split=dataset.idx_split)

    elif dataset_name == 'tau3mu':
        dataset = Tau3Mu(os.path.join(data_dir,dataset_name), data_config=data_config, seed=seed)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, idx_split=dataset.idx_split)

    elif dataset_name == 'synmol':
        dataset = SynMol(os.path.join(data_dir,dataset_name), data_config=data_config, seed=seed)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, idx_split=dataset.idx_split)

    elif dataset_name == 'plbind':
        dataset = PLBind(os.path.join(data_dir,dataset_name), data_config=data_config, n_jobs=32, debug=False)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, idx_split=dataset.idx_split, dataset_name=dataset_name)

    return loaders, test_set, dataset


def get_loaders_and_test_set(batch_size, dataset, idx_split, dataset_name=None):
    follow_batch = None if dataset_name != 'plbind' else ['x_lig']
    train_loader = DataLoader(dataset[idx_split["train"]], batch_size=batch_size, shuffle=True, follow_batch=follow_batch)
    valid_loader = DataLoader(dataset[idx_split["valid"]], batch_size=batch_size, shuffle=False, follow_batch=follow_batch)
    test_loader = DataLoader(dataset[idx_split["test"]], batch_size=batch_size, shuffle=False, follow_batch=follow_batch)

    test_set = dataset.copy(idx_split["test"])  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set
