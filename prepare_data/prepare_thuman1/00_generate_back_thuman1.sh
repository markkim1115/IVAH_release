#!/bin/bash

# Specify the path to your dataset directory
DATASET_DIR="/media/cv1/T7/THuman_MPS_NeRF/nerf_data_" 

# Iterate through each subdirectory in the dataset directory
for dir in "$DATASET_DIR"/*/ ; do
    
    subject_name=$(basename $dir)
    echo "Subject name: $subject_name"
    python prepare_data/prepare_thuman/00_generate_back_view.py --subject $subject_name
done