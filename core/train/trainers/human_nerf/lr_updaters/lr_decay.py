
from configs import cfg

def update_lr(optimizer, iter_step):
    decay_rate = cfg.train.lr_decay_rate
    decay_steps = cfg.train.lr_decay_steps

    if iter_step % decay_steps == 0:
        for param_group in optimizer.param_groups:
            new_lrate = cfg.train.lr * decay_rate
            param_group['lr'] = new_lrate
