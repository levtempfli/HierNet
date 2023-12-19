# Status: Complete
# Description: Random splitting of test datasets to val+test split
import dataset_loader
import tensorflow_datasets as tfds
import tensorflow as tf
import random
import json

import ds_tinyimagenet


def split(builderClass, classes):
    _, ds_test = dataset_loader.loadTfdsDataset(builderClass)
    test = []
    for i in range(0,classes):
        print("Filtering:" + str(i) + "/" + str(classes))
        ids = list(ds_test
                      .filter(lambda e: e['label']==tf.convert_to_tensor(i,dtype=tf.int64))
                      .map(lambda e:e['id']))
        ids = list(map(lambda a:tf.get_static_value(a).decode(),ids))
        test.append(ids)

    val = []
    for i in range(0,classes):
        print("Sampling:"+str(i)+"/"+str(classes))
        val.append(random.sample(test[i],len(test[i])//2))
        test[i]=list(filter(lambda e: e not in val[i],test[i]))

    val = [item for sublist in val for item in sublist]
    test = [item for sublist in test for item in sublist]

    split = {"val":val,"test":test}
    with open("ds_split_"+builderClass.name+".json", "w") as outfile:
        json.dump(split, outfile)

if __name__ == '__main__':
    split(builderClass=tfds.image.Cifar10, classes=10)
    split(builderClass=tfds.image.Cifar100, classes=100)
    split(builderClass=ds_tinyimagenet.TinyImagenet, classes=200)
