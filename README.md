## Customization

* For demo refer: `scripts/search_pic.sh`
* For CBIR report: `[reports/README.md](reports/README.md)`
* vector length is: vgg: 4096, resnet, maskrcnnlite: 2048 (explain)


**Changes / Contributions:**
* `demo2.py` - minor changes based on `demo.py`
* `scripts/` added
* `reports/` added
* `search.py` added
* `vector_search/vector_search.py` - key customization on model loading
  ```python
  ## To increase/set the GPU utilization
  import tensorflow as tf
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

  ##Added changes to load_headless_pretrained_model() to load 3 kinds of models
   print ("Input integers for which model to load. 1-Pretrained VGG16, 2-Pretrained ResNet50, 3-Resnet50")
    n1=input("Enter numbers 1/2/3")
    n=int(n1)
    if n==1:
        pretrained_vgg16 = VGG16(weights='imagenet', include_top=True)
        model = Model(inputs=pretrained_vgg16.input,
                 outputs=pretrained_vgg16.get_layer('fc2').output)

    if n==2:
        Pretrained_resn_model = ResNet50(weights='imagenet',include_top=True)
        model=Model(inputs=Pretrained_resn_model.inputs, outputs=resn_model.layers[-2].output)

    if n==3:
        model = load_mrcnnlite_model()


  ##load_mrcnnlite_model() loads the CNN used for classification in Mask R-CNN model
  import vector_search.maskrcnnlite as modellib

    MRCNN_ROOT = os.path.abspath("/home/intern/Documents/ai-ml-dl/external/MRCNN/Mask_RCNN")
    MODEL_DIR = os.path.join(MRCNN_ROOT, "logs")
    MRCNN_MODEL_PATH = os.path.join(MRCNN_ROOT, "mask_rcnn_coco.h5")
    print("MRCNN_MODEL_PATH: {}".format(MRCNN_MODEL_PATH))
    
    

    model = modellib.MaskRCNNLite(mode="inference", model_dir=MODEL_DIR)
    model.load_weights(MRCNN_MODEL_PATH, by_name=True)
    
  ```
  
  ```
  maskrcnnlite.py contains the coded architecture of the ResNet50 model we need for both Mask R-CNN and our Semantic Search 
  ```
* For MASK_RCNN integrations
  * core mrcnn code, credits:
  ```
  @misc{matterport_maskrcnn_2017,
    title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
    author={Waleed Abdulla},
    year={2017},
    publisher={Github},
    journal={GitHub repository},
    howpublished={\url{https://github.com/matterport/Mask_RCNN}},
  }
  ```
  * `vector_search/maskrcnnlite.py` - integrating/loading mask_rcnn model file
  * `mrcnnload.py` - for quick testing of maskrcnn model loding

**TODO**
* Put the custom dataset reference, on how to download
  - if possible give the script to download
  - instructions on where to store this dataset for demo


---
# Semantic Search
![Preview](https://github.com/hundredblocks/semantic-search/blob/master/assets/image_search_cover.jpeg)

This repository contains a barebones implementation of a semantic search engine. 
The implementation is based on leveraging pre-trained embeddings from VGG16 (trained on Imagenet), and GloVe (trained on Wikipedia).

It allows you to:
- Find similar images to an input image
- Find similar words to an input word
- Search through images using any word
- Generate tags for any image

See examples of usage by following along on this [notebook](http://insight.streamlit.io/0.13.3-8ErS/index.html?id=QAKzY9mLjr4WbTCgxz3XBX).
Read more details abbout why and how you would use this in this blog [post](https://blog.insightdatascience.com/the-unreasonable-effectiveness-of-deep-learning-representations-4ce83fc663cf).


## Setup
Clone the repository locally and create a virtual environment (conda example below):
```
conda create -n semantic_search python=3.5 -y
source activate semantic_search
cd semantic_search
pip install -r requirements.txt
```

If you intend to use text, download pre-trained GloVe vectors (we suggest to use the length 300 vectors):
```
curl -LO http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mkdir models
mkdir models/glove.6B
mv glove.6B.300d.txt models/glove.6B/
```

Download an example image dataset by using:
```
mkdir dataset
python downloader.py
mv dataset/diningtable dataset/dining_table
mv dataset/pottedplant dataset/potted_plant
mv dataset/tvmonitor dataset/tv_monitor
```
_Credit to Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier for the dataset_

### Running the pipeline end to end
Here is an example Streamlit tutorial for running the pipeline end to end!
```
python demo.py \
  --features_path feat_4096 \
  --file_mapping_path index_4096 \
  --model_path my_model.hdf5 \
  --custom_features_path feat_300 \
  --custom_features_file_mapping_path index_300 \
  --search_key 872 \
  --train_model True \
  --generate_image_features True \
  --generate_custom_features True \
  --training_epochs 1 \
  --glove_model_path models/glove.6B \
  --data_path dataset

```

### Usage
To make full use of this repository, feel free to import the vector_search package in your project. For added convenience, 
a few functions are exposed through a command line API. They are documented below. 

### Using pre-trained models for image search

#### Search for a similar image to an input image
First, you need to index your images:
```
python search.py \
  --index_folder dataset \
  --features_path feat_4096 \
  --file_mapping index_4096 \
  --index_boolean True \
  --features_from_new_model_boolean False
```

Then, you can search through your images using this index:
```
python search.py \
  --input_image dataset/cat/2008_001335.jpg \
  --features_path feat_4096 \
  --file_mapping index_4096 \
  --index_boolean False \
  --features_from_new_model_boolean False
```

### Training a custom model to map images to words
After you've downloaded the pascal dataset, and placed vectors in `models/golve.6B`
We recommond first training for 2 epochs to evluate performance. Each epoch is around 20 minutes on CPU. Full training on this dataset is around 50 epochs. 
```
python train.py \
  --model_save_path my_model.hdf5 \
  --checkpoint_path checkpoint.hdf5 \
  --glove_path models/glove.6B \
  --dataset_path dataset \
  --num_epochs 30
```

#### Index your images
Index the image using the custom trained model to file to not repeatedly do this operation in the future
```
python search.py \
  --index_folder dataset \
  --features_path feat_300 \
  --file_mapping index_300 \
  --model_path my_model.hdf5 \
  --index_boolean True \
  --features_from_new_model_boolean True \
  --glove_path models/glove.6B
```
#### Search for an image using image
```
python search.py \
  --input_image dataset/cat/2008_001335.jpg \
  --features_path feat_300 \
  --file_mapping index_300 \
  --model_path my_model.hdf5 \
  --index_boolean False \
  --features_from_new_model_boolean True \
  --glove_path models/glove.6B
```  

#### Search for an image using words
```
python search.py \
  --input_word cat \
  --features_path feat_300 \
  --file_mapping index_300 \
  --model_path my_model.hdf5 \
  --index_boolean False \
  --features_from_new_model_boolean True \
  --glove_path models/glove.6B
  ```

### Running the demo
After training and indexing the model, you can run the demo:
```
python demo.py \
  --features_path feat_4096 \
  --file_mapping_path index_4096 \
  --model_path my_model.hdf5 \
  --custom_features_path feat_300 \
  --custom_features_file_mapping_path index_300 \
  --search_key 872 \
  --train_model False \
  --generate_image_features False \
  --generate_custom_features False 
```

## Creating a custom dataset
Image dataset must be of the format below if you would like to import your own:
```
dataset/
|
|--- class_0/
|      |-------image_0
|      |-------image_1
|      ...
|
|      |-------image_n
|--- class_1/
|     ...
|  
|--- class_n/
```
Each class name should be one word in the english language, or multiple words separated by "_". 
In our example dataset for example, we rename the "diningtable" folder to dining'_'table.
