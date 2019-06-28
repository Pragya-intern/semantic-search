#!/bin/bash


modelname=$1
echo "Model supported: vgg, resnet, maskrcnnlite"

if [ -z ${modelname} ];then
  echo "Model name not give, defaulting to vgg."
  modelname="vgg"
fi

basepath=${HOME}/Documents/ai-ml-dl/external/semantic-search
echo "basepath: $basepath"
# ls -ltr $basepath/search.py
dataset_path=$basepath/dataset
echo "dataset_path: $dataset_path"


## vector length is:
## vgg: 4096
## resnet, maskrcnnlite: 2048

python $basepath/search.py \
  --input_image $dataset_path/llama/image_0046.jpg \
  --features_path feat-${modelname} \
  --file_mapping index-${modelname} \
  --index_boolean False \
  --features_from_new_model_boolean False

# python $basepath/search.py \
#   --input_image $dataset_path/llama/image_0046.jpg \
#   --features_path resn_feat \
#   --file_mapping resn_index \
#   --index_boolean False \
#   --features_from_new_model_boolean False
