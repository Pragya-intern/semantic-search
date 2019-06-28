#!/bin/bash

modelname=$1
echo "Model supported: vgg, resnet, maskrcnnlite"

if [ -z ${modelname} ];then
  echo "Model name not give, defaulting to vgg."
  modelname="vgg"
fi

## vector length is:
## vgg: 4096
## resnet, maskrcnnlite: 2048

basepath=${HOME}/Documents/ai-ml-dl/external/semantic-search
echo "basepath: $basepath"
# ls -ltr $basepath/search.py
dataset_path=$basepath/dataset
echo "dataset_path: $dataset_path"

python $basepath/search.py \
  --index_folder $dataset_path \
  --features_path feat-${modelname} \
  --file_mapping index-${modelname} \
  --index_boolean True \
  --features_from_new_model_boolean False
