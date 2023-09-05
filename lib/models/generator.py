import tensorflow as tf
from tensorflow.keras import Sequential, Model, layers


def get_generator() -> Model:
    generator_model = Sequential(name="generator")

    generator_model.add(layers.Dense(8 * 8 * 256, input_shape=(250,), use_bias=False))
    # generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU())
    assert generator_model.output_shape == (None, 16384)

    generator_model.add(layers.Reshape((8, 8, 256)))
    assert generator_model.output_shape == (None, 8, 8, 256)

    generator_model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert generator_model.output_shape == (None, 8, 8, 128)
    generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU())

    generator_model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert generator_model.output_shape == (None, 16, 16, 64)
    generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU())

    generator_model.add(
        layers.Conv2DTranspose(
            32, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert generator_model.output_shape == (None, 32, 32, 32)
    generator_model.add(layers.BatchNormalization())
    generator_model.add(layers.LeakyReLU())

    generator_model.add(
        layers.Conv2DTranspose(
            3,
            (5, 5),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.tanh,
        )
    )
    assert generator_model.output_shape == (None, 64, 64, 3)

    return generator_model


if __name__ == "__main__":
    generator_model = get_generator()
    generator_model.build()
    tf.keras.utils.plot_model(
        generator_model,
        to_file="generator.png",
        show_shapes=True,
    )
