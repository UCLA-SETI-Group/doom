""" A collection of models that can be used for signal classification.

This module is currently not meant to be run as a script.

Python Version
--------------
Requires Python 3
    Tested with Python 3.7


References
----------
[1] Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras,
    and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent
    Systems. O'Reilly Media, 2019.


Authors
-------
|    Paul Pinchuk (ppinchuk@physics.ucla.edu)


Jean-Luc Margot UCLA SETI Group.
University of California, Los Angeles.
Copyright 2021. All rights reserved.
"""

import warnings
import tensorflow as tf
from model_utils import SqueezeAndExcitationUnit, insert_layer, Conv2D, ResidualUnit, seti_input_layers, name_model

warnings.filterwarnings('ignore')


def standard_ResNet34(
        include_top=True, weights=None, input_tensor=None,
        input_shape=None, pooling=None, classes=1000, **__):
    """ Build a ResNet34 Keras Model.

    Parameters
    ----------
    include_top : bool, optional
        Flag indicating whether to include the fully-connected layer at
        the top of the network.
    weights : path-like, optional
        Either `None` (random initialization) or the path to the
        weights file to be loaded.
    input_tensor : Tensor object, optional
        Optional Keras tensor (i.e. output of layers.Input()) to
        use as image input for the model.
    input_shape : tuple, optional
        Optional shape tuple, only to be specified if include_top is
        `False` (otherwise the input shape has to be (224, 224, 3)
        (with 'channels_last' data format) or (3, 224, 224)
        (with 'channels_first' data format). It should have exactly 3
        input channels, and width and height should be no smaller than 32.
        E.g. (200, 200, 3) would be one valid value.
    pooling : {'avg', 'max', `None`}, optional
        Optional pooling mode for feature extraction when
        `include_top` is `False`.
            - `None` means that the output of the model will be the
            4D tensor output of the last convolutional block.
            - 'avg' means that global average pooling will be applied to
            the output of the last convolutional block, and thus the output
            of the model will be a 2D tensor.
            - 'max' means that global max pooling will be applied.
    classes : int, optional
        Optional number of classes to classify images into, only to be
        specified if `include_top` is `True`, and if no weights argument
        is specified.

    Returns
    -------
    `tf.keras.Model`
        A Keras model instance.

    """

    if input_tensor is None:
        input_tensor = tf.keras.layers.Input(shape=input_shape)
        conv1 = Conv2D(filters=64, kernel_size=7, strides=2)(input_tensor)
    else:
        input_tensor, out = input_tensor
        conv1 = Conv2D(filters=64, kernel_size=7, strides=2)(out)

    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.Activation('relu')(bn1)
    out = tf.keras.layers.MaxPool2D(
        pool_size=3, strides=2, padding='same'
    )(act1)
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        out = ResidualUnit(
            filters=filters, strides=1 if filters == prev_filters else 2
        )(out)
        prev_filters = filters

    if include_top:
        gap = tf.keras.layers.GlobalAvgPool2D()(out)
        flat = tf.keras.layers.Flatten()(gap)
        output = tf.keras.layers.Dense(
            units=classes, activation='softmax'
        )(flat)
        model = tf.keras.Model(inputs=[input_tensor], outputs=[output])
    else:
        if pooling is None:
            model = tf.keras.Model(inputs=[input_tensor], outputs=[out])
        elif pooling == 'avg':
            gap = tf.keras.layers.GlobalAvgPool2D()(out)
            model = tf.keras.Model(inputs=[input_tensor], outputs=[gap])
        elif pooling == 'max':
            gmp = tf.keras.layers.GlobalMaxPool2D(out)
            model = tf.keras.Model(inputs=[input_tensor], outputs=[gmp])
        else:
            raise ValueError(f"Invalid `pooling` argument: {pooling!r}")

    if weights:
        model.load_weights(weights)

    return model


# noinspection PyPep8Naming
def base_seti_ResNet(input_layer, version=50, **kwargs):
    seti_input = seti_input_layers(
        input_layer, return_input=version == 34,
        **kwargs
    )

    model_kwargs = {
        'weights': None,
        'input_tensor': seti_input,
        'include_top': False,
        'pooling': 'avg',
    }

    if version == 34:
        resNet = standard_ResNet34(**model_kwargs)
    elif version == 50:
        resNet = tf.keras.applications.resnet.ResNet50(**model_kwargs)
    elif version == 101:
        resNet = tf.keras.applications.resnet.ResNet101(**model_kwargs)
    elif version == 152:
        resNet = tf.keras.applications.resnet.ResNet152(**model_kwargs)
    else:
        raise ValueError(
            f"Version number must be one of: {{34, 50, 101, 152}}! "
            f"Passed in value: {version}")

    return resNet


def base_seti_VGG(input_layer, version=16, **kwargs):
    seti_input = seti_input_layers(
        input_layer,
        **kwargs
    )
    model_kwargs = {
        'weights': None,
        'input_tensor': seti_input,
        'include_top': False
    }

    if version == 16:
        vgg = tf.keras.applications.vgg16.VGG16(**model_kwargs)
    elif version == 19:
        vgg = tf.keras.applications.vgg19.VGG19(**model_kwargs)
    else:
        raise ValueError(
            f"Version number must be either 16 or 19! "
            f"Passed in value: {version}"
        )

    flat = tf.keras.layers.Flatten()(vgg.output)
    fc1 = tf.keras.layers.Dense(4096, activation='relu')(flat)
    fc2 = tf.keras.layers.Dense(4096, activation='relu')(fc1)
    vgg = tf.keras.Model(inputs=vgg.input, outputs=fc2)

    return vgg


def base_seti_Xception(input_layer, **kwargs):
    seti_input = seti_input_layers(
        input_layer,
        **kwargs
    )

    model_kwargs = {
        'weights': None,
        'input_tensor': seti_input,
        'include_top': False,
        'pooling': 'avg'
    }

    xception = tf.keras.applications.Xception(**model_kwargs)
    return xception


def standard_seti_model(
        model_factory,
        model_name=None,
        top_layers=None,
        **factory_kwargs
):
    model = model_factory([225, 225, 2], **factory_kwargs)
    top_layers = top_layers or []
    top_layers += [tf.keras.layers.Dense(
        units=1, activation='sigmoid', name='prediction'
    )]
    output = model.output
    for layer in top_layers:
        output = layer(output)
    model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[output],
        name=model_name
    )
    return model


def se_layer_factory(old_layer):
    return SqueezeAndExcitationUnit(
        n_chan=old_layer.output.shape[-1], 
        ratio=14
    )


def lr_func(lr0, s):
    def exponential_decay(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay


def get_seti_model(model_name, dropout_rate, bn_axis=3):
    seti_model = standard_seti_model(
        base_seti_Xception,
        model_name=model_name,
        include_bn=True,
        bn_axis=bn_axis,
        top_layers=[tf.keras.layers.Dropout(rate=dropout_rate)],
    )

    seti_model = insert_layer(
        seti_model,
        layer_regex='block[5-9]_sepconv3_bn|block1[012]_sepconv3_bn',
        new_layer_factory=se_layer_factory,
        position='after'
    )

    return seti_model
