from numpy import iterable
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
import torch
from torch import device, nn
import os
from transformers import AutoTokenizer, BertModel, T5Model, LlamaForSequenceClassification
import torch
import pickle
from tqdm import tqdm
from torch.nn import functional as F

class PLM(nn.Module):
    def __init__(self , config: Config) -> None:
        super().__init__()
        self.config = config
        self.cache = None
        self.backbone = ['dnn']
        self.embedding_layer = None
        self.i_embedding_layer = None
        self.forward = self.FeatureInteraction

    def load_feature_map(self):
        path = './MetaData/feature_index'
        self.feature_map = {}
        with open(path) as f:
            idx  = 0
            arr = f.readlines()
            while idx < len(arr):
                cot = arr[idx].strip().split('|')
                if len(cot) < 3:
                    cot = cot + arr[idx + 1].strip().split('|')
                    idx += 1
                idx += 1
                assert len(cot) >= 3
                xx = cot 
                field = xx[0]
                internal_id = xx.pop()
                feature = '|'.join(xx[1:])
                assert len(feature) > 0
                feature = feature.replace("raw_feat_","")
                if field not in self.feature_map:
                    self.feature_map[field] = {}
                self.feature_map[field][internal_id] = feature

    def textualization(self, data, open_field = ['category', 'description', 'title']):
        #[b, f]
        def parse(field, data):
            if not iterable(data):
                return self.feature_map[field][str(data)]
            else:
                res = ""
                for dd in data:
                    if int(dd.item()) == 0:
                        break
                    res += parse(field, int(dd.item())) + " "
                return res
        res = []
        for field, content in data.items(): 
            if field in open_field:
                content = list(map(lambda x: f"" + parse(field, x), content))
                res.append(content)
        return list(map(  lambda x: ' '.join(x) + ' ',zip(*res) ))
    
    def get_text(self, sample):
        if not hasattr(self, 'feature_map'):
            self.load_feature_map()
        return self.textualization(sample)
    
    def batch_encoding(self, input_content):
        start_idx = 0
        step = 1024
        res = []
        while start_idx < len(input_content):
            res_input = input_content[start_idx: min(start_idx + step, len(input_content))]
            input_ids = self.tokenizer(res_input, return_tensors="pt", padding= True, truncation= True, max_length = 80).input_ids.to(self.config.device)
            decoder_input_ids = self.tokenizer(res_input, return_tensors="pt", padding= True, truncation= True, max_length = 80).input_ids.to(self.config.device)
            decoder_input_ids = self.plm_encoder._shift_right(decoder_input_ids)
            outputs = self.plm_encoder.cuda()(input_ids=input_ids, decoder_input_ids=decoder_input_ids).last_hidden_state    
            outputs = torch.mean(outputs, dim = 1)
            res.append(outputs)
            start_idx += step
        res = torch.cat(res, dim = 0)
        return res

    def encoding(self):
        self.cache = {}
        dataset = self.config.dataset
        path = f'./MetaData/{self.config.dataset}_T5base_feature_index_encoding'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.tokenizer =  AutoTokenizer.from_pretrained("t5-small")
            self.plm_encoder = T5Model.from_pretrained("t5-small")
            self.plm_encoder.requires_grad_(False)
            self.load_feature_map()
            for sample in tqdm(dataset.train + dataset.val + dataset.test):
                input_content = self.textualization(sample)
                ids = sample['id']
                self.cache[ids] = self.batch_encoding(input_content).cpu()
            with open(path, 'wb') as f:
                obj = pickle.dumps(self.cache)
                f.write(obj)
            del self.plm_encoder

    def P(self, sparse_input):
        if not hasattr(self, 'tokenizer'):
            self.tokenizer =  AutoTokenizer.from_pretrained("google/flan-t5-small")
            self.plm_encoder = T5Model.from_pretrained("google/flan-t5-small")
            self.plm_encoder.requires_grad_(False)
            self.load_feature_map()
        input_content = self.textualization(sparse_input)
        return self.batch_encoding(input_content).cpu()

    def FeatureInteraction(self, sparse_input):
        if self.cache is None:
            self.encoding()
        ids = sparse_input['id']
        datas = self.cache[ids].to(self.config.device)
        self.logits = datas
        self.output = (self.logits)
        return self.output

    def RegularLoss(self , weight):
        return 0