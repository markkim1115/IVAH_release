import imp
from configs import cfg

def _query_trainer():
    module = cfg.trainer_module
    trainer_path = module.replace(".", "/") + ".py"
    trainer = imp.load_source(module, trainer_path).Trainer
    return trainer

def create_trainer(network, optimizer):
    Trainer = _query_trainer()
    return Trainer(network, optimizer)

def create_progress(fast_validation=False, test_mode=False):
    module = cfg.progress_module
    progress_path = module.replace(".", "/") + ".py"
    progress_renderer = imp.load_source(module, progress_path).Progress_Renderer(fast_validation, test_mode)
    return progress_renderer

def create_lr_updater():
    module = cfg.lr_updater_module
    lr_updater_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, lr_updater_path).update_lr

def create_optimizer(network, lr):
    module = cfg.optimizer_module
    optimizer_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, optimizer_path).get_optimizer(network, lr)