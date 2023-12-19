# Status: Complete
# Description:  Autohiernet with Resnet backbone for CIFAR100
#               Autohiernet = the hierarchy is constructed automatically from the confusion probabilities of the baseline model
# Accuracy:
#   -resnet_n=9,layer_split=14,Aadd=6,pretrain=no,transfer=yes,max_group=10,min_conn=0.005:     Cond:97.49/70.69/70.49, Route:96.94/69.99/70.11, A_pred:96.98/82.27/82.25
#   -resnet_n=9,layer_split=16,Aadd=4,pretrain=no,transfer=yes,max_group=10,min_conn=0.005:     Cond:97.36/71.43/69.31, Route:96.73/71.13/68.68, A_pred:96.77/83.09/80.84
#   -resnet_n=9,layer_split=18,Aadd=2,pretrain=no,transfer=yes,max_group=10,min_conn=0.005:     Cond:97.20/70.03/70.12, Route:96.61/69.57/69.57, A_pred:96.67/82.27/81.31
#   -resnet_n=9,layer_split=20,Aadd=0,pretrain=no,transfer=yes,max_group=10,min_conn=0.005:     Cond:96.03/69.61/68.28, Route:95.29/69.29/67.94, A_pred:95.42/83.05/82.03

#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=50,min_conn=0.05:       Cond:93.15/69.87/67.92, Route:92.64/69.65/67.88, A_pred:92.68/77.62/76.84
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=50,min_conn=0.025:      Cond:95.28/70.53/68.91, Route:94.53/70.35/68.40, A_pred:94.58/81.21/79.32
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=10,min_conn=0.02:       Cond:95.51/69.59/68.87, Route:94.74/69.09/68.44, A_pred:94.78/80.31/80.18
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=20,min_conn=0.015:      Cond:96.64/70.55/69.71, Route:95.94/69.89/69.25, A_pred:96.01/83.77/82.99
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=50,min_conn=0.015:      Cond:96.69/70.43/69.63, Route:95.98/69.81/69.19, A_pred:96.06/83.03/83.45
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=10,min_conn=0.01:       Cond:96.11/70.73/69.57, Route:95.34/70.51/68.99, A_pred:95.41/83.23/81.45
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=10,min_conn=0.005:      Cond:96.28/70.41/68.44, Route:95.52/69.77/68.00, A_pred:95.57/82.41/80.50
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=50,min_conn=0.01:       Cond:97.85/71.76/70.19, Route:97.39/71.23/69.63, A_pred:97.48/87.30/85.59
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=50,min_conn=0.0075:     Cond:98.63/71.65/70.65, Route:98.28/71.31/70.45, A_pred:98.49/90.73/90.28
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=20,min_conn=0.005:      Cond:97.41/70.35/69.45, Route:96.80/70.01/68.87, A_pred:96.88/85.28/84.17
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=yes,max_group=50,min_conn=0.005:      Cond:98.85/70.65/70.83, Route:98.65/70.43/70.51, A_pred:99.11/92.55/92.28

#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=no,transfer=no,max_group=50,min_conn=0.0075:      Cond:98.00/69.51/69.07, Route:97.67/69.13/68.62, A_pred:98.11/89.56/89.18
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=yes,transfer=yes,max_group=50,min_conn=0.0075:    Cond:98.87/70.47/69.67, Route:98.56/70.11/69.61, A_pred:98.81/90.06/89.78, A_predPT:98.53/89.90
#   -resnet_n=5,layer_split=9,Aadd=3,pretrain=yes,transfer=no,max_group=50,min_conn=0.0075:     Cond:98.40/70.63/69.75, Route:98.14/70.43/69.45, A_pred:98.65/90.52/89.84, A_predPT:98.16/89.42

#   -resnet_n=3,layer_split=5,Aadd=2,pretrain=no,transfer=yes,max_group=50,min_conn=0.0075:     Cond:94.63/69.05/68.89, Route:93.22/68.31/68.08, A_pred:93.54/86.76/86.23
#   -resnet_n=3,layer_split=5,Aadd=2,pretrain=no,transfer=yes,max_group=50,min_conn=0.025:      Cond:89.12/68.11/66.36, Route:87.71/67.11/65.76, A_pred:87.81/79.01/78.54
#   -resnet_n=7,layer_split=12,Aadd=4,pretrain=no,transfer=yes,max_group=50,min_conn=0.0075:    Cond:98.71/71.59/71.25, Route:98.34/71.21/70.75, A_pred:98.42/88.38/88.40
#   -resnet_n=7,layer_split=12,Aadd=4,pretrain=no,transfer=yes,max_group=50,min_conn=0.025:     Cond:95.78/71.05/68.93, Route:95.22/70.73/68.70, A_pred:95.26/80.71/78.98
#   -resnet_n=9,layer_split=16,Aadd=5,pretrain=no,transfer=yes,max_group=50,min_conn=0.0075:    Cond:99.52/71.81/72.15, Route:99.41/71.47/72.01, A_pred:99.49/90.54/90.60
#   -resnet_n=9,layer_split=16,Aadd=5,pretrain=no,transfer=yes,max_group=50,min_conn=0.025:     Cond:97.92/71.15/69.63, Route:97.56/70.73/69.09, A_pred:97.60/80.93/80.04
# Parent: hiernetV3.1_cifar10.py, resnet_cifar100.py
import json
import os
import argparse
import datetime
import re
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

from resnet_cifar100 import ResNet
import grouper
from autohiernet_resnet_cifar100_data import get_data
from autohiernet_resnet_cifar100_model import AutoHiernet
from autohiernet_resnet_cifar100_transfer import transfer

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
        res_model = ResNet(args.resnet_n)
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
