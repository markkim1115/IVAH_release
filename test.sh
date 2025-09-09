CUDA_VISIBLE_DEVICES=0 python test.py \
    --cfg 'experiments/renderpeople/full_model/2024-07-25-15-28-45_config.yaml' \
    --ckpt_path 'experiments/renderpeople/full_model/iter_800000.tar' \
    --test \
    
    ## If you want to evaluate using the estimated SMPL test set, add the following flag
    # --cliff_estimated_smpl_test \
    
    ## If you want to evaluate on novel pose synthesis test, add the following flag
    # --novel_pose_test