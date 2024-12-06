#!/bin/bash
dataset_name=$1
video_folder=$2
output_folder=$3


# NOTE: make sure to create xclip environment, see README
# Activate env
source .venv_xclip/bin/activate

# Get embeddings of videos
python scripts/xclip/xclip_embeddings.py --video_folder ${video_folder} --output_folder ${output_folder}

# Fit xgboost classification with the embeddings
python scripts/train_classification_heads.py --output_folder ${output_folder} --dataset_name ${dataset_name}

# Evaluate workflow
python scripts/evaluate_workflow.py --output_folder ${output_folder} --dataset_name ${dataset_name}