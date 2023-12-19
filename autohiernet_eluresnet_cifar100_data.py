# Status: Incomplete

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np

import dataset_loader

def get_data(group):
    ds_train, ds_val, ds_test = dataset_loader.loadTfdsDatasetWithValTest(tfds.image.Cifar100)

    ds_train = ds_train.map(lambda example: (tf.cast(example['image'], tf.float32) / 255.0, example['label']))
    ds_val = ds_val.map(lambda example: (tf.cast(example['image'], tf.float32) / 255.0, example['label']))
    ds_test = ds_test.map(lambda example: (tf.cast(example['image'], tf.float32) / 255.0, example['label']))

    total_count, per_pixel_sum = ds_train.reduce((np.float32(0), tf.zeros((32, 32, 3))),
                                                 lambda prev, curr: (prev[0] + 1.0, prev[1] + curr[0]))
    per_pixel_mean = per_pixel_sum / total_count

    img_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomTranslation(height_factor=0.125, width_factor=0.125, fill_mode="constant",
                                           fill_value=0.5)
        ]
    )
    ds_train = ds_train.map(lambda img, label: (img_augmentation(img, training=True), label))
    ds_val = ds_val.map(lambda img, label: (img, label))
    ds_test = ds_test.map(lambda img, label: (img, label))

    ds_train = ds_train.map(lambda img, label: (img - per_pixel_mean, label))
    ds_val = ds_val.map(lambda img, label: (img - per_pixel_mean, label))
    ds_test = ds_test.map(lambda img, label: (img - per_pixel_mean, label))

    relabel_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant([i[0] for l in group for i in l], dtype=tf.int64),
            values=tf.constant([i[2] for l in group for i in l], dtype=tf.int64)
        ),
        default_value=tf.constant(-1, dtype=tf.int64)
    )
    ds_train = ds_train.map(lambda img, label: (img, relabel_table.lookup(label)))
    ds_val = ds_val.map(lambda img, label: (img, relabel_table.lookup(label)))
    ds_test = ds_test.map(lambda img, label: (img, relabel_table.lookup(label)))

    # RelableA
    A_relable_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant([i[2] for l in group for i in l], dtype=tf.int64),
            values=tf.constant([i[3] for l in group for i in l], dtype=tf.int64)
        ),
        default_value=tf.constant(-1, dtype=tf.int64)
    )
    ds_train = ds_train.map(lambda img, label: (img, label, A_relable_table.lookup(label)))
    ds_val = ds_val.map(lambda img, label: (img, label, A_relable_table.lookup(label)))
    ds_test = ds_test.map(lambda img, label: (img, label, A_relable_table.lookup(label)))

    ds_train = ds_train.map(lambda img, label, labelA: (img, tf.one_hot(label, 100), tf.one_hot(labelA, len(group))))
    ds_val = ds_val.map(lambda img, label, labelA: (img, tf.one_hot(label, 100), tf.one_hot(labelA, len(group))))
    ds_test = ds_test.map(lambda img, label, labelA: (img, tf.one_hot(label, 100), tf.one_hot(labelA, len(group))))

    ds_train = ds_train.map(lambda img, label, labelA: (img, {"cond": label, "route": label, "A_pred": labelA}))
    ds_val = ds_val.map(lambda img, label, labelA: (img, {"cond": label, "route": label, "A_pred": labelA}))
    ds_test = ds_test.map(lambda img, label, labelA: (img, {"cond": label, "route": label, "A_pred": labelA}))

    ds_train_batched = ds_train.shuffle(5000).batch(128, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    ds_val_batched = ds_val.shuffle(5000).batch(128, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    ds_test_batched = ds_test.shuffle(5000).batch(128, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return ds_train_batched, ds_val_batched, ds_test_batched, ds_val
