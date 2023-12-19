# Status: Incomplete
import json
import os
import argparse
import datetime
import re
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

from eluresnet_cifar100 import EluResNet
import grouper
from autohiernet_eluresnet_cifar100_data import get_data
from autohiernet_eluresnet_cifar100_model import AutoHiernet
from autohiernet_eluresnet_cifar100_transfer import transfer

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors and warnings by default

parser = argparse.ArgumentParser()
parser.add_argument("--resnet_n", type=int, help="n from Resnet paper.", required=True)
parser.add_argument("--layer_split", type=int, help="Where to split the resnet for the root vs branches (This is the last layer included in root)", required=True)
parser.add_argument("--pretrain", type=str, help="Pretrain the first node", required=True,
                    choices=["no", "yes"])
parser.add_argument("--transfer", type=str, help="Transfer Learning from ResNet model", required=True,
                    choices=["no", "yes"])
parser.add_argument("--Aadd", type=int, help="Additional layers for A block", required=True)
parser.add_argument("--ref_logdir", type=str, help="The logdir path of the reference resnet model", required=True)
parser.add_argument("--max_group", type=int, help="Maximum group size", required=True)
parser.add_argument("--min_conn", type=float, help="Minimum connection strength", required=True)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def main(args):
    group = grouper.group(args.ref_logdir, args.max_group, args.min_conn)

    ds_train_batched, ds_val_batched, ds_test_batched, ds_val = get_data(group)

    model = AutoHiernet(args.resnet_n, args.Aadd, args.layer_split, group)

    os.makedirs(args.logdir)
    with open(args.logdir + '/grouping-' + str(args.max_group) + '-' + str(args.min_conn) + '.json', 'w') as of:
        of.write(json.dumps(group, indent=2))

    if args.transfer == 'yes':
        res_model = EluResNet(args.resnet_n)
        res_model.load_weights(args.ref_logdir + '/model.h5')
        transfer(model.layers[0], res_model.layers[0], set(()))

    learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
        [2000, 34000, 50000], [0.01, 0.1, 0.01, 0.001]
    )
    weight_decay = keras.optimizers.schedules.PiecewiseConstantDecay(
        [2000, 34000, 50000], [1e-5, 1e-4, 1e-5, 1e-6]
    )

    if args.pretrain != 'no':

        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir + '-pretrain', histogram_freq=1, update_freq=100,
                                                     profile_batch=0)
        model.compile(
            optimizer=tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=learning_rate, momentum=0.9,
                                          nesterov=False),
            loss={"A_pred": tf.losses.CategoricalCrossentropy()},
            metrics={"A_pred": tf.metrics.CategoricalAccuracy("A_pred_accuracy")},
        )

        model.fit(x=ds_train_batched, epochs=200, validation_data=ds_val_batched, callbacks=[tb_callback],
                  use_multiprocessing=True,
                  workers=args.threads)

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    model.compile(
        optimizer=tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=learning_rate, momentum=0.9,
                                      nesterov=False),
        loss={"cond": tf.losses.CategoricalCrossentropy()},
        metrics={"cond": [tf.metrics.CategoricalAccuracy("cond_top1"),
                          tf.keras.metrics.TopKCategoricalAccuracy(k=5,name="cond_top5")],
                 "route": [tf.metrics.CategoricalAccuracy("route_top1"),
                          tf.keras.metrics.TopKCategoricalAccuracy(k=5,name="route_top5")],
                 "A_pred": tf.metrics.CategoricalAccuracy("A_pred_accuracy")},
    )

    model.fit(x=ds_train_batched, epochs=200, validation_data=ds_val_batched, callbacks=[tb_callback],
              use_multiprocessing=True,
              workers=args.threads)

    test_evaluation = model.evaluate(ds_test_batched, return_dict=True, use_multiprocessing=True, workers=args.threads)
    with open(args.logdir + "/test_accuracy.txt", "w") as f:
        for metric in test_evaluation.items():
            f.write(metric[0] + " : " + str(metric[1]) + '\n')

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
                                   ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key),
                                                            value if key != 'ref_logdir' else re.sub(
                                                                ".*(\\d\\d\\d\\d-\\d\\d-\\d\\d_\\d\\d\\d\\d\\d\\d).*",
                                                                r"\1", value)) for key, value in
                                             sorted(vars(args).items())))
                               ))

    main(args)
