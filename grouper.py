# Status: Complete
# Description: Groups the CIFAR 10, 100 or TinyImagenet categories using the probability matrix of a trained model
#   -Can only be used on models with original label ordering
#   -The connection between two classes are the sum of their connections in the probability matrix
#   -Initially every category is a separate group
#   -Groups until it's not possible to group more
#   -Every iteration merges the first 2 groups with the largest connections between each other, that satisfy the restrictions: max group size, min connection strenght
#   -A group's connection to a class is the average connection from the group to that particular class
#   -Similarly a connection between two groups is the average of all the connections between them
import argparse
import numpy as np
import json

import ds_tinyimagenet
import resnet_cifar10
import resnet_cifar100

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, help="Logdir of the model to group", required=True)
parser.add_argument("--max_group", type=int, help="Maximum group size", required=True)
parser.add_argument("--min_conn", type=float, help="Minimum connection strength", required=True)


def group(logdir, max_group_size, min_connection):
    prob_matrix = np.load(logdir + '/prob.npy')
    n = prob_matrix.shape[0]
    if n == 10:
        labels = resnet_cifar10.labels
    elif n == 100:
        labels = resnet_cifar100.labels
    elif n == 200:
        labels = ds_tinyimagenet.labels
    else:
        raise ValueError("Input probability matrix size should be the size of Cifar10/Cifar100/TinyImagenet (10/100/200)")

    top = []
    for i in range(0, n):
        top.append([i])

    def dist(c1: list, c2: list):
        n = 0
        conn = 0
        for e1 in c1:
            for e2 in c2:
                conn += prob_matrix[e1][e2] + prob_matrix[e2][e1]
                n += 1
        conn /= n
        return conn

    no_more_possible = False
    while not no_more_possible:
        distances = []
        for i in range(0, len(top)):
            for j in range(i + 1, len(top)):
                distances.append(((top[i], top[j]), dist(top[i], top[j])))

        bests = sorted(distances, key=lambda x: x[1], reverse=True)

        no_more_possible = True
        for best in bests:
            if len(best[0][0]) + len(best[0][1]) <= max_group_size and best[1] > min_connection:
                no_more_possible = False
                top.remove(best[0][0])
                top.remove(best[0][1])
                top.append(best[0][0] + best[0][1])
                break

    total_count = 0
    group_count = 0
    for group in top:
        for i in range(0, len(group)):
            group[i] = (group[i], labels[group[i]], total_count, group_count)
            total_count += 1
        group_count += 1

    return top


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    top = group(args.logdir, args.max_group, args.min_conn)
    with open(args.logdir + '/grouping-' + str(args.max_group) + '-' + str(args.min_conn) + '.json', 'w') as of:
        of.write(json.dumps(top, indent=2))
