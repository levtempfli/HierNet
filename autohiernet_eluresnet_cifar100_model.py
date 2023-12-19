# Status: Incomplete

import tensorflow as tf
import tensorflow.keras as keras


class AutoHiernet(keras.Model):
    class ResidualBlock(tf.Module):
        def __init__(self, filters: int, down_sample: bool, name_initial: str):
            super().__init__()
            self.filters = filters
            self.down_sample = down_sample
            self.name_initial = name_initial

        def __call__(self, x):
            out = x

            out = keras.layers.Conv2D(filters=self.filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1) if not self.down_sample else (2, 2),
                                      padding="same",
                                      name=self.name_initial + "_conv1",
                                      use_bias=False,
                                      kernel_initializer=tf.keras.initializers.HeNormal)(out)
            out = keras.layers.ELU()(out)

            out = keras.layers.Conv2D(filters=self.filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      name=self.name_initial + "_conv2",
                                      use_bias=False,
                                      kernel_initializer=tf.keras.initializers.HeNormal)(out)
            out = keras.layers.BatchNormalization(name=self.name_initial + "_bn2")(out)

            if self.down_sample:
                residual = keras.layers.Conv2D(filters=self.filters,
                                               kernel_size=(1, 1),
                                               strides=(2, 2),
                                               padding="same",
                                               name=self.name_initial + "_resConv",
                                               use_bias=False,
                                               kernel_initializer=tf.keras.initializers.HeNormal)(x)
                residual = tf.keras.layers.BatchNormalization(name=self.name_initial + "_resBn")(residual)
            else:
                residual = x

            out = out + residual
            return out

    def get_late_node_block(self, A_out, n, resnet_n, name_initial: str, resnet_config, layer_split):
        X_out = A_out
        for i in range(layer_split, len(resnet_config)):
            filters, downsample = resnet_config[i]
            X_out = self.ResidualBlock(filters, downsample, name_initial + "_RB" + str(i))(X_out)

        X_pred = keras.layers.ELU()(X_out)
        X_pred = keras.layers.GlobalAveragePooling2D()(X_pred)
        X_pred = keras.layers.Dense(units=n,
                                    activation=keras.activations.softmax,
                                    name=name_initial + "SM" + str(n))(X_pred)
        return X_pred

    def __init__(self, resnet_n, Aadd, layer_split, group):
        inputs = keras.layers.Input(shape=(32, 32, 3), dtype=tf.float32, name="input")

        resnet_config = self.get_resnet_layer_config(resnet_n)

        # Root node
        A_out = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same", name="A_initConv",
                                    use_bias=False,
                                    kernel_initializer=tf.keras.initializers.HeNormal)(inputs)
        A_out = keras.layers.BatchNormalization(name="A_initBn")(A_out)
        A_out = keras.layers.ELU()(A_out)
        for i in range(0, layer_split):
            filters, downsample = resnet_config[i]
            A_out = self.ResidualBlock(filters, downsample, "A_RB" + str(i))(A_out)

        A_pred = A_out
        if Aadd > 0:
            for i in range(0, Aadd):
                filters, downsample = resnet_config[i + layer_split]
                A_pred = self.ResidualBlock(filters, downsample, "A_PredRB" + str(i))(A_pred)

        A_pred = keras.layers.GlobalAveragePooling2D()(A_pred)
        A_pred = keras.layers.Dense(units=len(group), activation=keras.activations.softmax,
                                    name="A_SM" + str(len(group)))(A_pred)

        # Subnodes
        subnode_preds = []
        for g in group:
            pred = self.get_late_node_block(A_out, len(g), resnet_n,
                                            "N" + str(len(subnode_preds)), resnet_config,
                                            layer_split)
            subnode_preds.append(pred)

        # Conditional probability prediction
        subnode_preds_cond = []
        for i in range(0, len(subnode_preds)):
            pred = subnode_preds[i]
            subnode_preds_cond.append(tf.einsum('ij,i->ij', pred, A_pred[:, i]))

        cond_outputs = tf.concat(subnode_preds_cond, 1)

        # Prediction by routing
        A_pred_argmax = tf.argmax(A_pred, axis=1)
        A_pred_route = tf.one_hot(A_pred_argmax, len(group))

        subnode_preds_route = []
        for i in range(0, len(subnode_preds)):
            pred = subnode_preds[i]
            subnode_preds_route.append(tf.einsum('ij,i->ij', pred, A_pred_route[:, i]))

        route_outputs = tf.concat(subnode_preds_route, 1)

        super().__init__(inputs=inputs,
                         outputs={**{"cond": cond_outputs,
                                     "route": route_outputs,
                                     "A_pred": A_pred},
                                  **{"N" + str(i) + "_pred": subnode_preds[i] for i in range(0, len(subnode_preds))}})

    def get_resnet_layer_config(self, resnet_n):
        config = []
        for i in range(0, resnet_n):
            config.append((16, False))
        config.append((32, True))
        for i in range(1, resnet_n):
            config.append((32, False))
        config.append((64, True))
        for i in range(1, resnet_n):
            config.append((64, False))
        return config
