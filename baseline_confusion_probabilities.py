# Status: Complete
# Description: Calculates and saves the confusion and confusion probability matrices of ResNet models
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def get_and_save_matrix(model: keras.Model, ds_val: tf.data.Dataset, n, model_folder, labels, img_size):
    ds_val = ds_val.map(lambda img, label: (img, label, model(tf.reshape(img, img_size))))

    confusion_matrix = np.zeros((n, n))
    prob_matrix = np.zeros((n, n))
    counter = np.zeros((n,))

    for ex in ds_val:
        current_pred = tf.argmax(ex[2], axis=1)[0]
        true = tf.argmax(ex[1])
        confusion_matrix[true][current_pred] += 1
        prob_matrix[true] += ex[2][0]
        counter[true] += 1

    prob_matrix /= counter[:, None]

    np.save(model_folder + '/prob.npy', prob_matrix)
    np.save(model_folder + '/conf.npy', confusion_matrix)

    def save_confustion(confusion_matrix, name):
        df_cm = pd.DataFrame(confusion_matrix,
                             index=[i for i in labels],
                             columns=[i for i in labels])
        plt.figure(figsize=(10, 7) if n <= 10 else (30, 21))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(model_folder + "/" + name + ".png")

    save_confustion(prob_matrix, "prob_matrix")
    save_confustion(confusion_matrix, "confusion_matrix")
