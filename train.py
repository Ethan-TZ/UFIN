from email.policy import strict
from Utils import Config
import numpy as np 
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Utils import Logger
import random as r

class Trainer():
    def __init__(self , filename) -> None:
        config = Config(filename)
        self.config = config
        self.ID = filename.split('.')[0]
        self.logger = Logger(config)
        self.interval = config.interval
        self.dataset = Dataset(config)
        self.config.dataset = self.dataset
        self.savedpath = config.savedpath
        config.device = torch.device("cuda")
        if os.environ["CUDA_VISIBLE_DEVICES"] != "":
            self.model = eval(f"{config.model}")(config).cuda()
        else:
            self.model = eval(f"{config.model}")(config)
        self.params, self.unparams = [], []

        for name, _ in self.model.named_parameters():
            if 'embedding' in name:
                self.unparams.append(_)
                self.params.append(_)
                continue
    
            if 'c_dnn' in name or 'lam' in name or 'tha' in name:#or 'embedding' in name:
                self.unparams.append(_)
            else:
                self.params.append(_)

        self.tune = True
        self.optimizer =  torch.optim.Adam(self.model.parameters(), lr=config.learning_rate , weight_decay=config.weight_decay)
        self.interval_cur = self.switch_interval = 300
        
        self.loss_fn = nn.BCELoss()
        self.best_auc = 0.
        self.epoch = 0
        self.early_stop_cnt = config.early_stop
        self.config = config
        self.draw_interval = len(self.dataset.train) // config.draw_loss_points
        self.has_live = True
        self.k_interval = 200
        self.epoch = 0
        print(config.dataset_name)
        if hasattr(config , 'pretrain'):
            self.savedpath = config.pretrain
            self.resume()

    def get_group_parameters(self, lr1, lr2):
        params = list(self.model.named_parameters())

        param_group = [
            {'params':[p for n,p in params if 'c_dnn' in n],'lr': lr1, 'weight_decay': 0.},
            {'params':[p for n,p in params if 'c_dnn' not in n],'lr': lr2, 'weight_decay': 1e-5}
        ]
        return param_group

    @property
    def current_state(self):
        return {
                'optimizer': self.optimizer.state_dict(), 
                'model': self.model.state_dict() , 
                'early_stop_cnt': self.early_stop_cnt , 
                'best_auc':self.best_auc,
                'epoch':self.step
                }
    
    def resume(self):
        save_info = torch.load(self.savedpath)
        related_params= {k:v for k,v in save_info['model'].items() if 'fine_tune_adaptiver' not in k}
        self.model.load_state_dict(related_params, strict = False)
        self.epoch = save_info['epoch'] + 1
        self.best_auc = 0#save_info['best_auc']
        print("model loaded !")
    
    def run(self):
        self.writer = SummaryWriter(self.config.logdir)
        self.train_process()
        self.evaluation_process()
        self.writer.close()

    def train_process(self):
        for i in range(self.epoch , 1000):
            self.step = i
            self.interval_cur -= 1
            if self.interval_cur < -self.switch_interval:
                self.interval_cur = self.switch_interval
            self.train_epoch(self.optimizer)        
            if i % self.interval == 0:
                auc , logloss = self.test_epoch(self.dataset.val)
                self.logger.record(self.step , auc ,logloss , 'val')
                self.writer.add_scalars('VAL/AUC' , {self.ID : auc} , self.step)
                self.writer.add_scalars('VAL/LOGLOSS' , {self.ID : logloss} , self.step)
                if auc > self.best_auc:
                    self.has_live = True
                    
                    print('find a better model !')
                    self.best_auc = auc
                    self.early_stop_cnt = self.config.early_stop
                    torch.save(self.current_state , self.savedpath + '_best')
                else:
                    self.early_stop_cnt -= 1
                if self.early_stop_cnt == 0:
                    return
    
    def evaluation_process(self):
        saved_info = torch.load(self.savedpath + '_best')
        self.model.load_state_dict(saved_info['model'])
        auc , logloss = self.test_epoch(self.dataset.test)
        self.logger.record(self.step , auc ,logloss , 'test')
        print(f"test , auc: {auc} , logloss: {logloss}")
        self.writer.add_scalars('TEST/AUC' , {self.ID : auc} , 0 )
        self.writer.add_scalars('TEST/LOGLOSS' , {self.ID : logloss} , 0)

        with open('./metafile.out', 'a+') as f:
            f.write(str(self.model.__class__) + ' ' + str(auc) + ' ' + str(logloss) + ' \n')

    def train_epoch(self, optimizer):
        cnt = 0
        res = 0
        self.model.train()
        for fetch_data in tqdm(self.dataset.train) if self.config.verbose else self.dataset.train:
            cnt += 1
            optimizer.zero_grad()
            prediction = self.model(fetch_data)
            loss = self.loss_fn(prediction.squeeze(-1) , fetch_data['label'].squeeze(-1).cuda()) \
                + self.model.RegularLoss(weight = self.config.L2) \

            loss.backward()
            optimizer.step()
            res += loss.cpu().item()

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

    trainer = Trainer(args.config_files)
    trainer.run()
