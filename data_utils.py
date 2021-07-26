""" Provide functionality for data loading and pre-processing.

This module is currently not meant to be run as a script.

Python Version
--------------
Requires Python 3
    Tested with Python 3.6.5


Authors
-------
|    Paul Pinchuk (ppinchuk@physics.ucla.edu)


Jean-Luc Margot UCLA SETI Group.
University of California, Los Angeles.
Copyright 2021. All rights reserved.
"""

import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _parse_setiml_tfrecord(
        serialized,
        min_max_norm=False
):
    """ Parse a single seti tfrecord and return results in dictionary. """
    features = {
        'ID': tf.io.FixedLenFeature([], tf.int64),
        'MJD': tf.io.FixedLenFeature([], tf.string),
        'FREQ': tf.io.FixedLenFeature([], tf.string),
        'DFDT': tf.io.FixedLenFeature([], tf.string),
        'table': tf.io.FixedLenFeature([], tf.string),
        'db': tf.io.FixedLenFeature([], tf.string),
        'obs_year': tf.io.FixedLenFeature([], tf.int64),
        'NPIX': tf.io.FixedLenFeature([], tf.int64),
        'SSROW': tf.io.FixedLenFeature([], tf.int64),
        'SHIFT': tf.io.FixedLenFeature([], tf.int64),
        'OTHER_ID': tf.io.FixedLenFeature([], tf.int64),
        'FLIP': tf.io.FixedLenFeature([], tf.int64),
        'IMAGES': tf.io.FixedLenFeature([], tf.string),
        'LABEL': tf.io.FixedLenFeature([], tf.int64),
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(
        serialized=serialized, features=features
    )
    parsed_example['MJD'] = tf.strings.to_number(
        parsed_example['MJD'], out_type=tf.float64
    )
    parsed_example['DFDT'] = tf.strings.to_number(
        parsed_example['DFDT'], out_type=tf.float64
    )
    parsed_example['FREQ'] = tf.strings.to_number(
        parsed_example['FREQ'], out_type=tf.float64
    )

    # decode and reshape images
    parsed_example['IMAGES'] = tf.io.decode_raw(
        parsed_example['IMAGES'], tf.float32
    )
    parsed_example['IMAGES'] = tf.reshape(
        parsed_example['IMAGES'],
        [2, parsed_example['NPIX'], parsed_example['NPIX']]
    )
    parsed_example['IMAGES'] = tf.transpose(
        parsed_example['IMAGES'], [1, 2, 0]
    )

    if min_max_norm:
        min_vals = tf.math.reduce_min(parsed_example['IMAGES'], axis=(0, 1))
        parsed_example['IMAGES'] = parsed_example['IMAGES'] - min_vals
        max_vals = tf.math.reduce_max(parsed_example['IMAGES'], axis=(0, 1))
        parsed_example['IMAGES'] = parsed_example['IMAGES'] / max_vals * 255

    return parsed_example


def seti_image_dataset(
        filepaths,
        n_reads=100,
        shuffle_buffer_size=1024,
        repeat=1,
        batch_size=64,
        cache=None,
        prefetch_buffer_size=AUTOTUNE,
        n_read_threads=None,
        min_max_norm=False
):
    """ Create a setiml dataset from a collection of tfrecord files.

    Parameters
    ----------
    filepaths : iterable
        Iterable of path-like objects representing the tfrecord
        files to be used in the data set.
    n_reads : int, optional
        Number of tfrecord files to read simultaneously.
        If `None`, data will be read sequentially.
    shuffle_buffer_size : int, optional
        Length of buffer used for shuffling. This buffer will
        be filled up and then sampled from to simulate shuffling.
        If `None` or `False`, data will not be shuffled.
    repeat : int, optional
        Number of times to repeat the data. If `None` or `False`,
        data will not be repeated.
    batch_size : int, optional
        Number of samples per batch. If `None` or `False`,
        data will not be batched.
    cache : str or `None`, optional
        Path to cache file to be used to speed up pre processing.
        If `None`, no cache is used.
    prefetch_buffer_size : int, optional
        Length of the prefetching buffer. If `None` or `False`,
        data will not be prefeteched.
    n_read_threads : int, optional
        Number of threads to use in parallel to read in the data.
        If `None`,  a single thread is used.
    for_training : {'siamese', `True`, `False`}, optional
        Flag indicating the use-purpose for this dataset.
        If 'siamese' (str), the images will be split into
        individual outputs (i.e. the returned values will be
        (top_image, bottom_image, label)). If `True`, the
        images will be returned as a single tensor
        (i.e. (images, label)). If `False`, the images
        will be returned as a single tensor, and the returned
        dictionary will contain all stored meta info
        for each example.
    min_max_norm : bool, optional
        Option to apply min-max normalization to the
        Images (on a per-image basis).

    Returns
    -------
    `tf.data.Dataset`
        Dataset containing setiml signal data.

    """

    dataset = tf.data.TFRecordDataset(
        filenames=filepaths, num_parallel_reads=n_reads
    )
    dataset = dataset.map(
        partial(
            _parse_setiml_tfrecord,
            min_max_norm=min_max_norm
        ),
        num_parallel_calls=1 if n_read_threads is None else n_read_threads
    )

    # caches pre processing work for data sets that don't fit in memory
    if isinstance(cache, str):
        dataset = dataset.cache(cache)

    if shuffle_buffer_size:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    if repeat:
        dataset = dataset.repeat(repeat)  # Repeat as much as user wants
    if batch_size:
        dataset = dataset.batch(batch_size)

    if prefetch_buffer_size is not None:
        # `prefetch` lets the dataset fetch batches
        # in the background while the model is training.
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    return dataset
