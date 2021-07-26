""" SETI ML model generation utilities.


Notes
-----
    This module is not intended to be run as a script.

Authors
-------
|    Paul Pinchuk (ppinchuk@physics.ucla.edu)


Jean-Luc Margot UCLA SETI Group.
University of California, Los Angeles.
Copyright 2021. All rights reserved.
"""

import re
import tensorflow as tf
import tempfile
from functools import partial
from itertools import chain


Conv2D = partial(tf.keras.layers.Conv2D, padding='same', use_bias=False)


class MCDropout(tf.keras.layers.Dropout):
    """ A Dropout Layer that applies Monte Carlo Dropout to the input.

    References
    ----------
    [1] Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn,
        Keras, and TensorFlow: Concepts, Tools, and Techniques to Build
        Intelligent Systems. O'Reilly Media, 2019. pp. 368-370.

    """

    def call(self, inputs, **_):
        return super().call(inputs, training=True)


class ResidualUnit(tf.keras.layers.Layer):
    """ A single residual unit (building block of residual networks).

    Parameters
    ----------
    filters : int
        Number of filters to use in each convolutional layer.
    strides : int, optional
        Number of strides to use for each convolutional layer.
        Recommended to be either 1 or 2.
    activation : str, optional
        Name of activation function to use.
    **kwargs
        Keyword arguments for :cls:`tf.keras.layer.Layer` class.

    References
    ----------
    [1] Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn,
        Keras, and TensorFlow: Concepts, Tools, and Techniques to Build
        Intelligent Systems. O'Reilly Media, 2019. pp. 471-474, 478.

    """

    def __init__(self, filters, strides=1, activation='relu', **kwargs):

        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.filters = filters
        self.strides = strides
        self.main_layers = [
            Conv2D(filters, kernel_size=3, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            Conv2D(filters, kernel_size=3, strides=1),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs, **_):
        skip_inputs = inputs

        # apply main forward pass
        for layer in self.main_layers:
            inputs = layer(inputs)

        # perform skip connection processing, if needed
        for layer in self.skip_layers:
            skip_inputs = layer(skip_inputs)

        # noinspection PyCallingNonCallable
        return self.activation(inputs + skip_inputs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'filters': self.filters,
            'strides': self.strides,
            'activation': tf.keras.activations.serialize(self.activation),
        }


class SqueezeAndExcitationUnit(tf.keras.layers.Layer):
    """ A single squeeze-and-excitation unit.

    Parameters
    ----------
    n_chan : int
        Number of channels in the input to the SE unit.
    ratio : int, optional
        Ratio of units in the latent space compared to input channels.
        Recommended to be factor of `n_chan`.
    dense_act : str, optional
        Name of activation function to use for Dense layer.
    **kwargs
        Keyword arguments for :cls:`tf.keras.layer.Layer` class.

    References
    ----------
    [1] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks."
        Proceedings of the IEEE conference on computer vision
        and pattern recognition. 2018.
    [2] Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn,
        Keras, and TensorFlow: Concepts, Tools, and Techniques to Build
        Intelligent Systems. O'Reilly Media, 2019. pp. 476-478.
    [3] https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7

    """

    def __init__(self, n_chan, ratio=16, dense_act='relu', **kwargs):
        super().__init__(**kwargs)
        self.n_chan = n_chan
        self.ratio = ratio
        self.dense_act = dense_act
        self.main_layers = [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(n_chan // ratio, activation=dense_act),
            tf.keras.layers.Dense(n_chan, activation='sigmoid'),
        ]

    def call(self, inputs, **_):
        out = self.main_layers[0](inputs)
        for layer in self.main_layers[1:]:
            out = layer(out)

        return tf.keras.layers.Multiply()([inputs, out])

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'n_chan': self.n_chan,
            'ratio': self.ratio,
            'dense_act': self.dense_act
        }


def name_model(model, name):
    """ Name a keras model.

    Parameters
    ----------
    model : `tensorflow.keras.Model`
        Model to be renamed.
    name : str
        New name for model.

    Returns
    -------
    `tensorflow.keras.Model`
        Model identical to the `unnamed_model with the given name.

    """
    model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.outputs,
        name=name
    )
    return model


def seti_input_layers(
        input_layer,
        return_input=False,
        include_bn=True,
        bn_axis=None
):
    """

    Parameters
    ----------
    input_layer : tuple or `tensorflow.keras.layers.Layer` instance
        Either the deesired shape of the input tensor in tuple
        format or an instance of the desired input layer.
    return_input : bool, optional
        Option to return the input layer along with the
        output of the batch_normalization layer, if applicable.
    include_bn : bool, optional
        Option to include a batch normalization layer
        immediately after the input layer.
    bn_axis : int or tuple, optional
        Axis parameter to pass to Batch Normalization layer.
        If `None`, then axis=[1, 2].

    Returns
    -------
    outputs
        Either a tensor corresponding to the output
        of the input layers or the same tensor plus
        the input layer, if requested.

    """

    try:
        input_layer = tf.keras.layers.Input(shape=input_layer, name='inputs')
    except TypeError:
        # if this error is thrown, we assume
        # input layer is already a layer
        pass

    if include_bn:
        # this may actually not help
        # see https://towardsdatascience.com/how-to-potty-train-a-siamese-network-3df6ca5e44da
        if bn_axis is None:
            bn_axis = [1, 2]
        out = tf.keras.layers.BatchNormalization(axis=bn_axis)(input_layer)
    else:
        out = input_layer

    if return_input:
        return input_layer, out
    else:
        return out


def transfer_weights(old_model, new_model):
    """ Transfer the weights from one model to another layer by layer.

    For this function to work properly, the layer structure
    of both models should be identical. This is useful, for
    example, in cases where a dropout layer is replaced with
    a monte carlo dropout layer.

    Parameters
    ----------
    old_model, new_model : `tensorflow.keras.Model`
        `old_model` contains the weights to be transferred
        to `new_model`.

    """
    for new_layer, old_layer in zip(new_model.layers, old_model.layers):
        new_layer.set_weights(old_layer.get_weights())


def insert_layer(
        model,
        layer_regex,
        new_layer_factory,
        position='after',
        reset_model=True
):
    """ Insert new layers into an existing model.

    This implementation is a heavily modified version of
    the top StackOverflow answer in the reference link below.

    Parameters
    ----------
    model : `tensorflow.keras.Model`
        Model instance containing the original layers
        that should be modified in some way.
    layer_regex : str
        Regular expression used to match to layers
        for which the new layer insertion
        should happen around.
    new_layer_factory : callable
        A callable that takes the layer matching the
        `layer_regex` as input and outputs
        an *iterable* of layers to insert.
    position : {'before', 'after', 'replace'}, optional
        Flag indicating the position of the inserted layer w.r.t
        the layer matching the `layer_regex`. Must be one of
        'before', 'after', or 'replace'.
    reset_model : bool, optional
        Option to save and immediately load model graph. This
        is highly recommended in order to avoid any problems
        when using this function multiple times.

    Returns
    -------
    `tensorflow.keras.Model`
        New model instance containing the inserted layers.

    References
    ----------
    [1] https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model

    """

    if position not in ('before', 'after', 'replace'):
        raise ValueError('position must be: before, after or replace')

    network_dict = convert_model_to_input_output_dict(model)

    tf.keras.backend.clear_session()

    model_outputs = []
    for layer in model.layers[1:]:

        x = _inputs_to_layer(layer, network_dict)

        if re.match(layer_regex, layer.name):
            new_layers = _insert_new_layer(layer, new_layer_factory, position)
            for new_layer in new_layers:
                x = new_layer(x)

        else:
            x = _recreate_layer(layer)(x)

        network_dict['output_tensor_of'][layer.name] = x

        if layer.name in model.output_names:
            model_outputs.append(x)

    model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model_outputs,
        name=model.name
    )

    if reset_model:
        with tempfile.TemporaryDirectory() as dir_name:
            model.save(dir_name)
            model = tf.keras.models.load_model(dir_name)

    return model


def convert_model_to_input_output_dict(model):
    """ Convert a model graph to a dictionary with input/output info.

    The dictionary will contain two pieces of info for each layer:
    the input layers and the output tensor.

    Parameters
    ----------
    model : tensorflow.keras.Model instance
        Model containing layers to analyze.

    Returns
    -------
    dict
        For every layer in the model, this dictionary contains the
        input layers ('input_layers_of') as well as the output tensor
        ('output_tensor_of').

    """

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            network_dict['input_layers_of'].setdefault(
                layer_name, []
            ).append(layer.name)

    network_dict['output_tensor_of'][model.layers[0].name] = model.input
    return network_dict


def _inputs_to_layer(layer, network_dict):
    """ Extract the inputs to `layer` from the network dict. """

    x = [
        network_dict['output_tensor_of'][layer_in]
        for layer_in in network_dict['input_layers_of'][layer.name]
    ]
    if len(x) == 1:
        x = x[0]
    return x


def _insert_new_layer(old_layer, new_layer_factory, position):
    """ Insert the new layer(s) in correct position w.r.t the old layer. """

    new_layers = _new_layers_as_iterable(new_layer_factory, old_layer)

    # Instead of re-using the old layer, we create a copy so that the
    # graph is clean
    layer = _recreate_layer(old_layer)

    if position == 'after':
        new_layers = chain([layer], new_layers)
    elif position == 'before':
        new_layers = chain(new_layers, [layer])

    return new_layers


def _recreate_layer(layer):
    layer_conf = layer.get_config()
    # the catch below is not fail-safe... If the user names the layer
    # something other than "Squeeze", this function will fail
    if 'squeeze' in layer_conf['name']:
        return SqueezeAndExcitationUnit(**layer_conf)
    return layer.__class__(**layer_conf)


def _new_layers_as_iterable(new_layer_factory, old_layer):
    """ Call the layer factory and return output as an iterable container. """

    new_layers = new_layer_factory(old_layer)
    try:
        iter(new_layers)
    except TypeError:
        new_layers = [new_layers]
    return new_layers


def __sequential_standard_ResNet34():
    """ Create a ResNet34 using `tf.keras.models.Sequential`. """

    model = tf.keras.models.Sequential()
    model.add(Conv2D(
        filters=64, kernel_size=7, strides=2, input_shape=[224, 224, 3]
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=3, strides=2, padding='same'
    ))
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        model.add(ResidualUnit(
            filters=filters, strides=1 if filters == prev_filters else 2
        ))
        prev_filters = filters
    model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    return model
