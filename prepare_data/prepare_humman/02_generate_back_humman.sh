#!/bin/bash

# Specify the path to your dataset directory
DATASET_DIR="/media/cv1/T7/HuMMaN/recon" 

for dir in "$DATASET_DIR"/*/ ; do
    subject_name=$(basename $dir)
    
    # if subject name starts with p0004, skip
    if echo $subject_name | grep -q "p0004"; then
        echo "Skipping $subject_name"
        continue
    fi

    # if subject name starts with p0005 and number is less than 595, skip
    if echo $subject_name | grep -q "p0005"; then
        prefix=$(echo ${subject_name} | cut -d'_' -f1 | cut -c2-)
        number=$(echo $prefix | sed 's/^0*//')
        
        if [ $number -lt 595 ]; then
            echo "Skipping $subject_name"
            continue
        fi
    fi
    
    echo "Subject name: $subject_name"
    python prepare_data/prepare_humman/02_generate_back_view.py --subject $subject_name
done

# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000480_a000793
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000502_a000041
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p100077_a000048
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000558_a000793
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p100045_a000224
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000579_a000396
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000474_a000048
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p100050_a001425
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000594_a001134
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000590_a000071
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000577_a000048
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p100047_a001425
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000486_a000793
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000584_a000048
# python prepare_data/prepare_humman/02_generate_back_view.py --subject p000579_a000048
