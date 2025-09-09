import numpy as np
from core.data import create_dataloader
from core.nets.create_network import create_network
from core.train.create_components import create_trainer, create_optimizer, create_progress
from configs import cfg
import torch
import random

def main():
    # Fix random seed for reproducibility
    random.seed(5000)
    np.random.seed(5000)
    torch.manual_seed(100000)
    torch.cuda.manual_seed(100000)

    cfg.evaluate = False
    model = create_network()
    optimizer = create_optimizer(model, cfg.train.lr)
    print("Loaded network, optimizer and scheduler")
    trainer = create_trainer(model, optimizer)
    print("Loaded trainer")
    progress_checker = create_progress(fast_validation=True, test_mode=False)
    progress_checker.set_device(trainer.device)
    full_tester = create_progress(fast_validation=False, test_mode=True)
    full_tester.set_device(trainer.device)
    
    # Set progress module for novel pose synthesis test
    cfg.novel_pose_test = True
    full_tester_novel_pose = create_progress(fast_validation=False, test_mode=True)
    full_tester_novel_pose.set_device(trainer.device)
    cfg.novel_pose_test = False
    print("Loaded Progress, Test module")

    train_loader = create_dataloader('train')
    print("Loaded Train loader")

    train_end = False
    
    while True:
        for batch_idx, batch_data in enumerate(train_loader):
            if trainer.iter > cfg.train.maxiter:
                train_end = True
                break
            trainer.train(train_loader, batch_idx, batch_data)
            if trainer.iter in [5000, 10000] or trainer.iter % cfg.train.val_iter == 0:
                is_reload_model = False
                is_reload_model = progress_checker.render_progress(trainer.network, trainer.iter, back_net=trainer.back_net)
                if is_reload_model:
                    trainer.reload_network()
                else:
                    trainer.save_ckpt(cfg.logdir, f'iter_{trainer.iter}')
                    trainer.save_ckpt(cfg.logdir, 'latest')
                    progress_checker.validate(trainer.network, trainer.iter, trainer.writer, False, back_net=trainer.back_net)
            
            if trainer.iter % cfg.train.maxiter == 0:
                full_tester.render_progress(trainer.network, trainer.iter, back_net=trainer.back_net)
                full_tester.validate(trainer.network, trainer.iter, None, True, back_net=trainer.back_net)
                full_tester_novel_pose.render_progress(trainer.network, trainer.iter, back_net=trainer.back_net)
                full_tester_novel_pose.validate(trainer.network, trainer.iter, None, True, back_net=trainer.back_net)

        if train_end:
            break
    
    print("Training complete")
    exit(0)
        
if __name__ == '__main__':
    main()
