#!/bin/bash

# Check that an argument was passed
if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

DATASET=$1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dpvo

# Run the Python attack
python demo.py \
  --imagedir=input_output/images/clean_frames_${DATASET} \
  --calib=input_output/calib/S_Fly.txt \
  --stride=1 \
  --name=clean_${DATASET} \
  --plot

python3 PGD_attack_final.py \
  --name ${DATASET} \
  --imagedir=input_output/images/clean_frames_${DATASET}

python demo.py \
  --imagedir=input_output/images/noised_frames_${DATASET} \
  --calib=input_output/calib/S_Fly.txt \
  --stride=1 \
  --name=noised_PGD_${DATASET} \
  --plot