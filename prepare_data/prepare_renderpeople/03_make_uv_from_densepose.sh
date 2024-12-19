#!/bin/bash

# Specify the path to your dataset directory
DATASET_DIR="/home/cv1/works/SingleHumanNeRF/dataset/RenderPeople" 
DETECTRON_PATH='/home/cv1/works/detectron2'

# Iterate through each subdirectory in the dataset directory
for dir in "$DATASET_DIR"/*/ ; do
    
    subject_name=$(basename $dir)
    echo "Subject name: $subject_name"
    python prepare_data/prepare_renderpeople/03_make_uv_from_densepose.py \
        --root $DATASET_DIR --subject $subject_name --detectron_path $DETECTRON_PATH \
        --extract_densepose --extract_partial_uv_textures
done