from email.policy import default
import pathlib
import yaml
import time
import sys
ROOT = pathlib.Path(__file__).parent.parent

class Config:
    def __init__(self , filename = None) -> None:
        cmd = self._load_cmd_line()
        self.readconfig('defaultruntime.yaml')
        self.readconfig(filename)
        for k, v in cmd.items():
            setattr(self, k, v)

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

    def _load_cmd_line(self):
        cmd_config_dict = dict()
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                try:
                    cmd_config_dict[cmd_arg_name] = float(cmd_arg_value)
                except:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        return cmd_config_dict

if __name__ == "__main__":
    xx = Config('default.yaml')
    print(xx)