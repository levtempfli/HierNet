# Status: Complete
# Description: Transfer learning for Autohiernet:Resnet:Cifar100
import tensorflow.keras as keras


def transfer(Hlayer, Rlayer, seen):
    if Hlayer in seen:
        return
    seen.add(Hlayer)

    Hlayer.set_weights(Rlayer.get_weights())

    for Hnext in map(lambda l: l.outbound_layer, Hlayer.outbound_nodes):
        if isinstance(Hnext, keras.layers.GlobalAveragePooling2D):
            continue

        Rnext = list(
            filter(lambda l: isinstance(l, Hnext.__class__) and (
                    not isinstance(l, keras.layers.Conv2D) or l.kernel_size == Hnext.kernel_size),
                   map(lambda n: n.outbound_layer, Rlayer.outbound_nodes)))
        if len(Rnext) != 1:
            raise "Cannot decide next node in transfer"

        transfer(Hnext, Rnext[0], seen)