# Status: Complete
# Decription: Universal loader of tf.Datasets, works with CIFAR10/100 and Tiny-ImageNet
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import json


def loadTfdsDataset(builderClass: tfds.core.GeneratorBasedBuilder.__class__):
    download_config = None
    data_dir = None
    download_dir = None
    if os.environ.get('TFENVIRONMENT') == 'server':
        download_config = tfds.download.DownloadConfig(
            extract_dir='/datasetsE/extract',
            manual_dir='/datasetsE',
        )
        download_dir = '/datasetsE'
        data_dir = '/home/datasets'

    builder: tfds.core.GeneratorBasedBuilder = builderClass(data_dir=data_dir)
    builder.download_and_prepare(download_config=download_config, download_dir=download_dir)

    return builder.as_dataset(split='train'), builder.as_dataset(split='test')


def loadTfdsDatasetWithValTest(builderClass: tfds.core.GeneratorBasedBuilder.__class__):
    ds_train, ds_test = loadTfdsDataset(builderClass)
    with open("ds_split_" + builderClass.name + ".json", "r") as infile:
        split = json.load(infile)

    which_split = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(split['val'] + split['test'], dtype=tf.string),
            values=tf.constant(['val' for _ in split['val']] + ['test' for _ in split['test']], dtype=tf.string)
        ),
        default_value=tf.constant('unkw', dtype=tf.string)
    )

    ds_val = ds_test.filter(lambda ex: which_split.lookup(ex['id']) == tf.convert_to_tensor('val'))
    ds_test = ds_test.filter(lambda ex: which_split.lookup(ex['id']) == tf.convert_to_tensor('test'))
    return ds_train, ds_val, ds_test
