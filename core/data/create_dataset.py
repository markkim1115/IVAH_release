import imp
import time
import numpy as np
import torch
from configs import cfg
import pdb

def _query_dataset(mode):
    module = cfg[mode].dataset_module
    module_path = module.replace(".", "/") + ".py"
    dataset = imp.load_source(module, module_path).Dataset
    return dataset

def set_mode_args(args:dict, mode):
    if mode == 'train':
        args.update({
            'keyfilter':cfg.train_keyfilter,
            'ray_shoot_mode': cfg.train.ray_shoot_mode
        })
    else:
        args.update({
            'keyfilter':cfg.test_keyfilter,
            'ray_shoot_mode':'image',
        })
    
    return args

def fetch_db_config(dbname):
    if dbname == 'thuman2':
        return 'dataset/thuman2'
    elif dbname == 'RenderPeople':
        return 'dataset/RenderPeople'
    elif dbname == 'thuman1':
        return 'dataset/thuman1_mpsnerf'
    elif dbname == 'humman':
        return 'dataset/humman'
    elif dbname == 'itw':
        return 'dataset/itw_data'

def create_dataset(mode='train', fast_validation=False):
    args = {}
    args['mode'] = mode
    args = set_mode_args(args, mode)
    
    args['dataset_path'] = fetch_db_config(cfg.db)

    # customize dataset arguments according to dataset type
    args['bgcolor'] = None if mode == 'train' else cfg.bgcolor
    args['target_view'] = '0'
    args['fast_validation'] = fast_validation
    args['novel_pose_test'] = cfg.novel_pose_test
    
    if mode == 'progress':
        args['maxframes'] = cfg.progress.maxframes
    
    if mode in ['freeview', 'tpose']:
        args['skip'] = cfg.render_skip
    
    dataset = _query_dataset(mode)
    dataset = dataset(**args)
    return dataset


def _worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))

def create_dataloader(mode='train', fast_validation=False):
    cfg_node = cfg[mode]
    batch_size = cfg_node.batch_size
    shuffle = cfg_node.shuffle
    drop_last = cfg_node.drop_last
    dataset = create_dataset(mode=mode, fast_validation=fast_validation)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=cfg.num_workers,
                                              worker_init_fn=_worker_init_fn)

    return data_loader
