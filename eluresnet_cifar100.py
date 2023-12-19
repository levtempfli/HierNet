# Status: Incomplete

import argparse
import dataset_loader
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import datetime
import re

import baseline_confusion_probabilities

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors and warnings by default

parser = argparse.ArgumentParser()
parser.add_argument("--resnet_n", type=int, help="n from Resnet paper.", required=True)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
          'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
          'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
          'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
          'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
          'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
          'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
          'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
          'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
          'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

##################################
class EluResNet(keras.Model):
    class ResidualBlock(tf.Module):
        def __init__(self, filters: int, down_sample: bool):
            super().__init__()
            self.filters = filters
            self.down_sample = down_sample

        def __call__(self, x):
            out = x

            out = keras.layers.Conv2D(filters=self.filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1) if not self.down_sample else (2, 2),
                                      padding="same",
                                      use_bias=False,
                                      kernel_initializer=tf.keras.initializers.HeNormal)(out)
            out = keras.layers.ELU()(out)

            out = keras.layers.Conv2D(filters=self.filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      use_bias=False,
                                      kernel_initializer=tf.keras.initializers.HeNormal)(out)
            out = keras.layers.BatchNormalization()(out)

            if self.down_sample:
                residual = keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(2, 2),
                                               padding="same",
                                               use_bias=False,
                                               kernel_initializer=tf.keras.initializers.HeNormal)(x)
                residual = tf.keras.layers.BatchNormalization()(residual)
            else:
                residual = x

            out = out + residual
            return out

    def __init__(self, resnet_n):
        inputs = keras.layers.Input(shape=(32, 32, 3), dtype=tf.float32)
        outputs = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
                                      kernel_initializer=tf.keras.initializers.HeNormal)(
            inputs)
        outputs = keras.layers.BatchNormalization()(outputs)
        outputs = keras.layers.ELU()(outputs)

        for _ in range(0, resnet_n):
            outputs = self.ResidualBlock(16, False)(outputs)

        outputs = self.ResidualBlock(32, True)(outputs)
        for _ in range(1, resnet_n):
            outputs = self.ResidualBlock(32, False)(outputs)

        outputs = self.ResidualBlock(64, True)(outputs)
        for _ in range(1, resnet_n):
            outputs = self.ResidualBlock(64, False)(outputs)

        outputs = keras.layers.ELU()(outputs)
        outputs = keras.layers.GlobalAveragePooling2D()(outputs)
        outputs = keras.layers.Dense(100, activation=tf.nn.softmax)(outputs)
        super().__init__(inputs, outputs)

############################################

def get_data():
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
    ds_train = ds_train.map(lambda img, label: (img_augmentation(img, training=True), tf.one_hot(label, 100)))
    ds_val = ds_val.map(lambda img, label: (img, tf.one_hot(label, 100)))
    ds_test = ds_test.map(lambda img, label: (img, tf.one_hot(label, 100)))

    ds_train = ds_train.map(lambda img, label: (img - per_pixel_mean, label))
    ds_val = ds_val.map(lambda img, label: (img - per_pixel_mean, label))
    ds_test = ds_test.map(lambda img, label: (img - per_pixel_mean, label))

    ds_train_batched = ds_train.shuffle(5000).batch(128, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    ds_val_batched = ds_val.shuffle(5000).batch(128, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    ds_test_batched = ds_test.shuffle(5000).batch(128, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return ds_train_batched, ds_val_batched, ds_test_batched, ds_val


def main(args, tb_callback):
    ds_train_batched, ds_val_batched, ds_test_batched, ds_val = get_data()

    model = EluResNet(args.resnet_n)

    learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
        [32000, 48000], [0.1, 0.01, 0.001]
    )
    weight_decay = keras.optimizers.schedules.PiecewiseConstantDecay(
        [32000, 48000], [1e-4, 1e-5, 1e-6]
    )
    if args.resnet_n >= 18:
        learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
            [2000, 34000, 50000], [0.01, 0.1, 0.01, 0.001]
        )
        weight_decay = keras.optimizers.schedules.PiecewiseConstantDecay(
            [2000, 34000, 50000], [1e-5, 1e-4, 1e-5, 1e-6]
        )

    model.compile(
        optimizer=tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=learning_rate, momentum=0.9,
                                      nesterov=False),
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=[tf.metrics.CategoricalAccuracy("accuracy"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5,name="top5")],
    )

    model.fit(x=ds_train_batched, epochs=200, validation_data=ds_val_batched, callbacks=[tb_callback],
              use_multiprocessing=True,
              workers=args.threads)

    test_evaluation = model.evaluate(ds_test_batched, return_dict=True, use_multiprocessing=True, workers=args.threads)
    with open(args.logdir + "/test_accuracy.txt", "w") as f:
        for metric in test_evaluation.items():
            f.write(metric[0] + " : " + str(metric[1]) + '\n')

    print('Calculating confusion...')
    baseline_confusion_probabilities.get_and_save_matrix(model, ds_val, 100, args.logdir, labels, (1, 32, 32, 3))

    model.save(args.logdir + '/model.h5')
    print('OK')

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("{}/{}".format("logs", os.path.basename(globals().get("__file__", "notebook"))),
                               "{}-{}".format(
                                   datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
                                   ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in
                                             sorted(vars(args).items())))
                               ))

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    main(args, tb_callback)
