import torch.optim as optim

from configs import cfg

_optimizers = {
    'adam': optim.Adam
}

def get_customized_lr_names():
    return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]

def get_optimizer(network, lr):
    optimizer = _optimizers[cfg.train.optimizer]
    if cfg.train.optimizer == 'adam':
        optimizer = optimizer(network.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        assert False, "Unsupported Optimizer."
        
    return optimizer
