load_feature_stastic= ['label', 'user_id', 'item_id', 'brand', 'title', 'description', 'category']
feature_defaults= [[0], [0], [0], [0], ['x'], ['x'], ['x']]
import pathlib
ROOT = pathlib.Path(__file__).parent
import torch
import numpy as np
import pickle
from tqdm import tqdm
import os


class Interaction():
    def __init__(self, data = None, phase = None) -> None:
        self._d = {} if data is None else data
        self.phase = phase
    
    def __add__(self, other):
        for k in other:
            x = other[k]
            # if not isinstance(x, list):
            #     x = x.tolist()
            if isinstance(x[0], bytes):
                x = list(map(lambda x: torch.Tensor(list(map(int,x.decode('utf-8').split(' '))))[:min(len(x),  32)], x))
            
            if isinstance(x, list):
                self._d[k] = self._d.get(k, []) + x
            elif isinstance(x, np.ndarray):
                self._d[k] = np.concatenate([self._d.get(k, np.array([])), x], axis = 0)
            elif isinstance(x, torch.Tensor):
                self._d[k] = torch.cat([self._d.get(k, type(x)([])), x], dim = 0)
        return self

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._d[idx]
        
        res = {}
        for k in self._d:
            res[k] = self._d[k][idx]
        return Interaction(res)
    
    def __len__(self):
        for k, v in self._d.items():
            return len(v)
    
    def __iter__(self):
        return iter(sorted(self._d.keys()))
    
    def __setitem__(self, k, v):
        self._d[k] = v
    
    def items(self):
        return self._d.items()
    
    def cache_load(self, path):
        for file in os.listdir(path):
            attr, phase = file.split('&')
            if self.phase == phase:
                with open(path + f'/{file}' , 'rb') as f:
                     self._d[attr] = pickle.load(f)
    
    def cache_save(self, path):
        for attr, value in self._d.items():
            with open(path + f'/{attr}&{self.phase}' , 'wb') as f:
                obj = pickle.dumps(value)
                f.write(obj)

def parse_file(filename, load_feature_stastic, feature_default):
    import tensorflow as tf

    dataset = tf.data.TextLineDataset(filename)
    
    def decoding(record , feature_name , feature_default):
        data = tf.io.decode_csv(record , feature_default)
        feature = dict( zip(feature_name , data) )
        if len(feature_name) > 7:
            feature['user_id'] = feature['user_id'] + 996786
            feature['item_id'] = feature['item_id'] + 2526990
        label = feature.pop('label')
        return feature , label

    dataset = dataset.map(lambda line : decoding(line , load_feature_stastic , feature_default) , num_parallel_calls = 10).batch(10240000)
    
    Data = []
    for data in tqdm(dataset.as_numpy_iterator()):
        record = data[0]
        record['label'] = data[1].astype(np.float32)
        Data.append(record)
    return Data

def loading(embeddings):
    for dataset_name in ['Books','Movies_and_TV', 'ML']:
        if dataset_name == 'ML':
            global load_feature_stastic
            global feature_defaults
            load_feature_stastic = ['user_id', 'item_id', 'label', 'weekday', 'hour', 'age', 'gender', 'occupation','zip_code', 'movie_title', 'release_year', 'genre']
            feature_defaults = [[0], [0], [0], [0], [0], [0], [0], [0], [0], ['x'], [0], ['x']]
        path = str(ROOT / 'DataSource' / f'{dataset_name}')
        train_file = parse_file(path +'_train' , load_feature_stastic, feature_defaults)
        val_file = parse_file(path  +'_val' , load_feature_stastic, feature_defaults)
        test_file = parse_file(path +'_test' , load_feature_stastic, feature_defaults)
        train= Interaction(phase='train')
        for record in tqdm(train_file):
            train = train + record

        for record in tqdm(val_file):
            train = train + record

        for record in tqdm(test_file):
            train = train + record

        items = train['item_id']

        path = str(ROOT / 'MetaData' / f'{dataset_name}_flant5small_feature_index_encoding')

        if os.path.exists(path):
            idx = 0
            iid = 0

        with open(path, 'rb') as f:
            buffer = pickle.load(f)

            while idx in buffer:
                cur = buffer[idx]
                lens = len(cur)
                item = items[iid: iid + lens]

                tar, indices = np.unique(item, return_index=True)
                v = cur[indices]
                with torch.no_grad():
                    embeddings.weight[tar.astype(np.int64)] = (v)

                idx += 1
                iid += lens
            
            del buffer
        print('loaded!')
        del train
        del train_file
        del val_file
        del test_file
    return embeddings