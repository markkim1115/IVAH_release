import os
import time
import shutil

from termcolor import colored
from configs import cfg, args
import pprint
import torch
from core.utils.file_util import save_dict_to_yaml

class Logger(object):
    r"""Write log messages to a file."""
    def __init__(self, path=None, clean_up=True, create_yaml=True):
        self.log_path = os.path.join(cfg.logdir, 'logs.txt') if path == None else path

        log_dir = os.path.dirname(self.log_path)
        if clean_up:
            if not cfg.resume and os.path.exists(log_dir):
                user_input = input(f"log dir \"{log_dir}\" exists. \nRemove? (y/n):")
                if user_input == 'y':
                    print(colored('remove contents of directory %s' % log_dir, 'red'))
                    os.system('rm -r %s/*' % log_dir)
                else:
                    print(colored('exit from the training.', 'red'))
                    exit(0)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.log = open(self.log_path, "a") if os.path.exists(self.log_path) and os.path.isfile(self.log_path) else open(self.log_path, "w")
        if create_yaml:
            exptime = time.strftime('%Y-%m-%d-%H-%M-%S')
            save_dict_to_yaml(cfg, os.path.join(cfg.logdir, f'{exptime}_config_raw.yaml'))
            shutil.copy(args.cfg, os.path.join(cfg.logdir, f'{exptime}_config.yaml'))

    def write(self, message):
        self.log.write(message+'\n')
        self.log.flush()

    def flush(self):
        pass

    def print_config(self):
        print("\n\n######################### CONFIG #########################\n")
        print(cfg)
        print("\n##########################################################\n\n")
    
    def write_config(self):
        self.log.write(f'GPU name -> {torch.cuda.get_device_name()}')
        self.log.write(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')
        self.log.write(pprint.pformat(cfg))