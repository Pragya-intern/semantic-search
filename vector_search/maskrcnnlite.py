from keras.models import Model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D


class MaskRCNNLite():
  
    def __init__(self, mode=None, config=None, model_dir=None):
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.keras_model = self.build(mode=mode, config=config)


    def build(self, mode=None, config=None):
        # Inputs
        # IMAGE_SHAPE
        # IMAGE_META_SIZE
        # NUM_CLASSES = 2
        # IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES
        # input_image_meta = Input(shape=[IMAGE_META_SIZE], name="input_image_meta")

        input_image = Input(shape=[None, None, 3], name="input_image")
        model = self.resnet_model(input_image)
        return model


    def resnet_model(self, input_image, architecture="resnet50", stage5=False, train_bn=False):
        """Build a ResNet graph.
            architecture: Can be resnet50 or resnet101
            stage5: Boolean. If False, stage5 of the network is not created
            train_bn: Boolean. Train or freeze Batch Norm layers
        """
        assert architecture in ["resnet50", "resnet101"]
        # Stage 1
        x = ZeroPadding2D((3, 3))(input_image)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
        x = BatchNormalization(name='bn_conv1')(x, training=train_bn)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        # Stage 2
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
        # Stage 3
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
        # Stage 4
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
        '''
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
        C4 = x
        # Stage 5
        if stage5:
            x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
            x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
            C5 = x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
       '''
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b', train_bn=train_bn)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c', train_bn=train_bn)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d', train_bn=train_bn)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e', train_bn=train_bn)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f', train_bn=train_bn)


        # Stage 5
        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)


        x = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(x)

        # output layer
        # x = Flatten()(x)

        model = Model(inputs=input_image, outputs=x, name='ResNet50')
        return model


    def identity_block(self, input_tensor, kernel_size, filters, stage, block,
                       use_bias=True, train_bn=True):
        """The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      use_bias=use_bias)(input_tensor)
        x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

        x = Add()([x, input_tensor])
        x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block,
                   strides=(2, 2), use_bias=True, train_bn=True):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                      '2c', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
        shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

        x = Add()([x, shortcut])
        x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x


    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()


    def predict(self, inputs):
      '''
      ## TODO: proably need to use some code from detect in original maskrcnn to re-shape the 
      ## inputs, currently going blind faith with default predict function of keras
      '''
      # self.detect(images, verbose=1)
      self.keras_model.predict(inputs)
      return
