# You can run each command separately to test the different pre-trained models.

# CUDA_VISIBLE_DEVICES=0 python test.py \
#     --cfg configs/human_nerf/in_the_wild/renderpeople_ckpt.yaml \
#     --ckpt_path experiments/renderpeople/full_model_fix/iter_800000.tar \
#     --test


# CUDA_VISIBLE_DEVICES=0 python test.py \
#     --cfg configs/human_nerf/in_the_wild/humman_ckpt.yaml \
#     --ckpt_path experiments/humman/full_model/iter_800000.tar \
#     --test

CUDA_VISIBLE_DEVICES=0 python test.py \
    --cfg configs/human_nerf/in_the_wild/humman_ckpt.yaml \
    --ckpt_path experiments/humman/full_model/iter_800000.tar \
    --test 
    
# CUDA_VISIBLE_DEVICES=0 python test.py \
#     --cfg configs/human_nerf/in_the_wild/thuman2_ckpt.yaml \
#     --ckpt_path experiments/thuman2/thuman2_full_model/iter_600000.tar \
#     --test