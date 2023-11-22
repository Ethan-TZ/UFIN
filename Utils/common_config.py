from email.policy import default
import pathlib
import yaml
import time
ROOT = pathlib.Path(__file__).parent.parent

class Config:
    def __init__(self , filename = None) -> None:
        self.readconfig('defaultruntime.yaml')
        self.readconfig(filename)

    def readconfig(self , filename) -> None:
        filepath = str(ROOT / 'RunTimeConf' / filename)
        self.logger_file = str(ROOT / 'RunLogger' / (filename+time.strftime("%d_%m_%Y_%H_%M_%S")))
        f = open(filepath , 'r', encoding='utf-8')
        desc = yaml.load(f.read(),Loader=yaml.FullLoader)
        f.close()
        for key , value in desc.items():
            setattr(self,key,value)

        self.savedpath = str(ROOT / 'Saved' / (desc['model'] + '_'.join(desc['dataset']) ))
        self.logdir = str(ROOT / 'Log')
        self.dataset_name = '_'.join(desc['dataset'])

if __name__ == "__main__":
    xx = Config('default.yaml')
    print(xx)