#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

DATASET=$1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dpvo

python demo.py --imagedir=input_output/images/clean_frames_${DATASET} --calib=input_output/calib/S_Fly.txt --stride=1 --name=square_${DATASET} --plot