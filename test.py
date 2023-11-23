from Model import PEPNet
from Model.PEPNet import *
from Model.RMoE import *
from Model.EulerNet import *
from Utils import Config
import numpy as np 
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm
from Utils import Logger
import random as r
from torch.nn import functional as F
import os
class Tester():
    def __init__(self , filename) -> None:
        config = Config(filename)
        self.config = config

        self.ID = filename.split('.')[0]
        self.logger = Logger(config)
        self.interval = config.interval
        self.dataset = Dataset(config)
        self.savedpath = config.savedpath
        self.config.dataset = self.dataset
        self.model = UFIN(config).cuda()
        
        print(config.dataset_name)
        self.savedpath = config.savedpath
        if os.path.exists(self.savedpath):
            save_info = torch.load(self.savedpath)
            related_params= {k:v for k,v in save_info['model'].items()}#if 'decoder' in k}
            self.model.load_state_dict(related_params, strict = False)
            print("model loaded !")
    
    def run(self):
        auc, logloss = self.test_epoch(self.dataset.test)
        print(auc,logloss)

    def test_epoch(self , datasource):
        with torch.no_grad():
            self.model.eval()
            val , truth = [] , []
            for fetch_data in tqdm(datasource) if self.config.verbose else datasource:
                prediction = self.model(fetch_data)
                val.append(prediction.cpu().numpy())
                truth.append(fetch_data['label'].numpy())

            y_hat = np.concatenate(val, axis=0).squeeze()
            y = np.concatenate(truth, axis=0).squeeze()
            auc = roc_auc_score(y, y_hat)
            logloss = - np.sum(y*np.log(y_hat + 1e-6) + (1-y)*np.log(1-y_hat+1e-6)) /len(y)
        return auc , logloss


if __name__ == '__main__':
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_files", help="path to load config", default="/home/data/tz/DAGFM_pytorch/RunTimeConf_Criteo/fibinet.yaml")
    parser.add_argument("--gpu", help="path to load config", default=0)
    args = parser.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torch import nn
    from Data.dataset import Dataset
    from Model import *
    setup_seed(2022)

    trainer = Tester(args.config_files)
    trainer.run()
