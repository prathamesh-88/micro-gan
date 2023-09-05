import tensorflow as tf
from tensorflow.keras import layers
from constants import global_constants


def get_normalized_dataSet(path="./dataset"):
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels=None,
        shuffle=True,
        image_size=(64, 64),
        batch_size=global_constants["BATCH_SIZE"],
    )
    rescaling_layer = layers.Rescaling(scale=1.0 / 127.5, offset=-1)
    dataset = dataset.map(lambda x: rescaling_layer(x))

    return dataset
