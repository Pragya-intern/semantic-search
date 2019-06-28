#!/bin/bash

basepath=${HOME}/Documents/ai-ml-dl/external/semantic-search
echo "basepath: $basepath"
# ls -ltr $basepath/search.py
dataset_path=$basepath
echo "dataset_path: $dataset_path"

python $basepath/search.py \
  --index_folder $dataset_path/dataset \
  --features_path feat_4096 \
  --file_mapping index_4096 \
  --index_boolean True \
  --features_from_new_model_boolean False
