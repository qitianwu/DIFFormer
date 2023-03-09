import numpy as np
import os
import torch
import pickle as pkl

from data_utils import rand_train_test_idx, class_rand_splits, to_sparse_tensor

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}

        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, sub_dataname=''):
    """ Loader for NCDataset 
        Returns NCDataset 
    """
    if dataname.startswith('mini'):
        dataset =load_mini_imagenet(data_dir)
    elif dataname.startswith('20news'):
        dataset=load_20news(data_dir)
    elif dataname=='stl10':
        dataset=load_stl10(data_dir)
    elif dataname=='cifar10':
        dataset=load_cifar10(data_dir)
    else:
        raise ValueError('Invalid dataname')
    return dataset

def load_mini_imagenet(data_dir):
    path = data_dir + '/mini_imagenet/mini_imagenet.pkl'

    dataset = NCDataset('mini_imagenet')

    data=pkl.load(open(path,'rb'))
    x_train=data['x_train']
    x_val=data['x_val']
    x_test=data['x_test']
    y_train=data['y_train']
    y_val=data['y_val']
    y_test=data['y_test']

    features=torch.cat((x_train,x_val,x_test),dim=0)
    labels=np.concatenate((y_train,y_val,y_test))
    num_nodes=features.shape[0]

    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(labels)
    return dataset

def load_20news(data_dir):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    path = os.path.join(data_dir, '20news')

    if os.path.exists(os.path.join(path, '20news.pkl')):
        data = pkl.load(open(os.path.join(path, '20news.pkl'), 'rb'))
    else:
        categories = ['alt.atheism',
                      'comp.sys.ibm.pc.hardware',
                      'misc.forsale',
                      'rec.autos',
                      'rec.sport.hockey',
                      'sci.crypt',
                      'sci.electronics',
                      'sci.med',
                      'sci.space',
                      'talk.politics.guns']
        data = fetch_20newsgroups(data_home=path, subset='all', categories=categories)

    vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
    X_counts = vectorizer.fit_transform(data.data).toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    features = transformer.fit_transform(X_counts).todense()
    features=torch.Tensor(features)
    y = data.target
    y=torch.LongTensor(y)
   
    num_nodes=features.shape[0]

    dataset = NCDataset('20news')
    dataset.graph={'edge_index': None,
            'edge_feat': None,
            'node_feat': features,
            'num_nodes': num_nodes}
    dataset.label=torch.LongTensor(y)

    return dataset

def load_stl10(data_dir):
    path=os.path.join(data_dir, 'stl10', 'features.pkl')
    dataset = NCDataset('stl10')
    data=pkl.load(open(path,'rb'))
    x_train,y_train,x_test,y_test=data['x_train'],data['y_train'],data['x_test'],data['y_test']
    x=np.concatenate((x_train,x_test),axis=0)
    y=np.concatenate((y_train,y_test))

    dataset.label=torch.LongTensor(y)
    x=torch.Tensor(x)
    num_nodes=x.shape[0]
    dataset.graph={'edge_index': None,
            'edge_feat': None,
            'node_feat': x,
            'num_nodes': num_nodes}
    return dataset

def load_cifar10(data_dir, num_image=15000):
    path=os.path.join(data_dir, 'cifar10', 'features.pkl')
    dataset = NCDataset('cifar10')
    data=pkl.load(open(path,'rb'))
    x_train,y_train,x_test,y_test=data['x_train'],data['y_train'],data['x_test'],data['y_test']
    x=np.concatenate((x_train,x_test),axis=0)
    y=np.concatenate((y_train,y_test))
    x=x[:num_image]
    y=y[:num_image]

    dataset.label=torch.LongTensor(y)
    x=torch.Tensor(x)
    num_nodes=x.shape[0]
    dataset.graph={'edge_index': None,
            'edge_feat': None,
            'node_feat': x,
            'num_nodes': num_nodes}
    return dataset

if __name__=='__main__':
    # load_stl10()
    load_cifar10()