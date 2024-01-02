from json import load
import pathlib
ROOT = pathlib.Path(__file__).parent.parent
import yaml
import torch
import numpy as np
import os
from Utils.common_config import Config
import pickle
from sklearn.metrics import log_loss, roc_auc_score

from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils
from Model import *
xx = {
    'Movies_and_TV' : 1,
    'Grocery_and_Gourmet_Food' : 2,
    'Electronics' : 3,
    'Books' : 4,
    'Musical_Instruments': 5,
    'Office_Products' : 0,
    'Toys_and_Games': 6,
    'ML': 6
}
enable = False
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
    
    def remove(self):
        for xx in list(self._d.keys()):
            if xx not in ['item_id', 'user_id', 'label', 'teacher_logits']:
                del self._d[xx]

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

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, config, dataset, shuffle = True) -> None:
        self.generator = torch.Generator()
        self.generator.manual_seed(config.seed)
        sample_space = np.arange(len(dataset))

        for k in dataset:
            v = dataset[k]
            dataset[k] = np.array(v) if isinstance(v, list) else v
        
        self._data = dataset._d

        super().__init__(
            dataset= sample_space,
            batch_size= config.batch_size,
            collate_fn= self.collate_fn,
            num_workers= config.num_works,
            shuffle= config.shuffle and shuffle,
            generator= self.generator,
        )

    def collate_fn(self, indices):
        res = {}
        for k in self._data.keys():
            d = self._data[k][indices]
            try:
                d = torch.Tensor(d)
            except:
                d = rnn_utils.pad_sequence(d, batch_first=True)
            
            res[k] = d
        if enable:
            res['encoding'] = Hook.llm_embedding(res['item_id'].long())
        return res

class Hook():
    def scene_hook(ccg, dataset_name):
        for se in ccg:
            idx = torch.ones_like(torch.Tensor(se._d['item_id'])) * xx[dataset_name]
            se._d['scene'] = idx

    def teacher_hook(config, ccg, dataset, model_name = 'EulerNet'):
        def assert_auc(logits, se):
            predict = torch.sigmoid(logits).numpy().squeeze()
            label = se['label'].squeeze()
            auc = roc_auc_score(label, predict)
            print(f'auc: {auc}')            
        teacher = eval(model_name)(config).cuda()
        save_info = torch.load(str(ROOT / 'Saved') +'/' + model_name+dataset+'_best')
        related_params= {k:v for k,v in save_info['model'].items() if 'i_embedding' not in k}#if 'decoder' in k}
        teacher.load_state_dict(related_params, strict = False)
        teacher.requires_grad_(False)
        teacher.eval()
        for se in ccg:
            loader = DataLoader(config, se, False)
            res = []
            with torch.no_grad():
                for data in tqdm(loader):
                    teacher(data)
                    res.append(teacher.logits.cpu())
            res = torch.cat(res, dim = 0)
            se._d['teacher_logits'] = res
            assert_auc(res, se)

    def language_hook(config, ccg, dataset):
        if len(dataset) ==  0:
            from torch import nn
            from get_embedding import loading
            Hook.llm_embedding = nn.Embedding(2530874, 512)
            Hook.llm_embedding.requires_grad_(False)
            loading(Hook.llm_embedding)
            return        
        plm_name = 'flant5small'
        pre_path = str(ROOT / 'MetaData')
        path = f'{pre_path}/{dataset}_{plm_name}_feature_index_encoding'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                cache = pickle.load(f)
            
            if isinstance(cache, dict):
                res = []
                idx = 0
                while idx in cache:
                    res.append(cache[idx])
                    idx += 1
                res = torch.cat(res, dim = 0)
            else:
                res = cache
            del cache

            st = 0
            for se in ccg:
                ed = st + len(se)
                se._d['sencoding'] = res[st:ed]
                st = ed
        else:
            config.device = torch.device("cuda")
            plm = PLM(config)
            total = {}
            idx = 0
            for se in ccg:
                loader = DataLoader(config, se, False)
                res = []
                with torch.no_grad():
                    for data in tqdm(loader):
                        codes = plm.P(data)
                        res.append(codes)
                        #total.append(codes)
                        total[idx] = codes
                        idx += 1
                res = torch.cat(res, dim = 0)
                se._d['encoding'] = res
            #total = torch.cat(total, dim = 0)
            with open(path, 'wb') as f:
                obj = pickle.dumps(total)
                f.write(obj)
                f.close()

class Dataset():
    def __init__(self , config : Config) -> None:
        train, val, test = Interaction(phase='train'), Interaction(phase='val'), Interaction(phase='test')
        feature_stastic = {}
        if enable:
            Hook.language_hook(config, [], [])
        for dataset in config.dataset:
            d_train, d_val, d_test, d_feature_stastic = self.load_single_dataset(dataset)
            config.feature_stastic = d_feature_stastic
            # Hook.language_hook(config, [d_train, d_val, d_test], dataset)
            if hasattr(config, 'distill_from'):
                Hook.teacher_hook(config, [d_train, d_val, d_test], dataset, config.distill_from)
            # d_train.remove()
            # d_val.remove()
            # d_test.remove() 

            Hook.scene_hook([d_train, d_val, d_test], dataset)
            train = train + d_train
            val = val + d_val
            test = test + d_test
            feature_stastic.update(d_feature_stastic)

        config.feature_stastic = feature_stastic
        self.train = DataLoader(config, train)
        self.val = DataLoader(config, val)
        self.test = DataLoader(config, test)
        
    def load_single_dataset(self, dataset_name):

        with open(str(ROOT / 'MetaData' / ('all_data.yaml' if dataset_name != 'ML' else 'ML.yaml') ) , 'r') as f:
            descb = yaml.load(f.read(),Loader=yaml.FullLoader)
            feature_stastic = descb['feature_stastic']
            feature_default = descb['feature_defaults']
            load_feature_stastic = descb['load_feature_stastic']

        train, val, test = Interaction(phase='train'), Interaction(phase='val'), Interaction(phase='test')

        train_file = self.parse_file(str(ROOT / 'DataSource' / dataset_name) +'_train' , load_feature_stastic, feature_default)
        val_file = self.parse_file(str(ROOT / 'DataSource' / dataset_name)  +'_val' , load_feature_stastic, feature_default)
        test_file = self.parse_file(str(ROOT / 'DataSource' / dataset_name) +'_test' , load_feature_stastic, feature_default)

        for record in tqdm(train_file):
            train = train + record

        for record in tqdm(val_file):
            val = val + record

        for record in tqdm(test_file):
            test = test + record
                
        return train, val, test, feature_stastic

    
    def parse_file(self, filename, load_feature_stastic, feature_default):
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
    
    def padding_sequence(self, data):
        data = data.tolist()
        data = list(map(lambda x: torch.Tensor(list(map(int,x.decode('utf-8').split(' '))))[:min(len(x),  80)], data))
        return rnn_utils.pad_sequence(data, batch_first=True)
