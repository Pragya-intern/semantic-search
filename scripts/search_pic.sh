#!/bin/bash

basepath=${HOME}/Documents/ai-ml-dl/external/semantic-search
echo "basepath: $basepath"
# ls -ltr $basepath/search.py
dataset_path=$basepath
echo "dataset_path: $dataset_path"

## VGGNet

## ResNet
python $basepath/search.py \
  --input_image $dataset_path/llama/image_0046.jpg \
  --features_path resn_feat \
  --file_mapping resn_index \
  --index_boolean False \
  --features_from_new_model_boolean False


## Mask_RCNN



# python search.py \
#   --input_image dataset/llama/image_0046.jpg \
#   --features_path resn_feat \
#   --file_mapping resn_index \
#   --index_boolean False \
#   --features_from_new_model_boolean False
