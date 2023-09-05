import tensorflow as tf
from tensorflow.keras import Sequential, Model, layers


# Discriminator model


def get_discriminator() -> Model:
    discriminator_model = Sequential(name="discriminator")

    discriminator_model.add(layers.Input((64, 64, 3)))

    discriminator_model.add(
        layers.Conv2D(64, kernel_size=10, strides=(1, 1), padding="same")
    )
    discriminator_model.add(layers.LeakyReLU())
    assert discriminator_model.output_shape == (None, 64, 64, 64)

    discriminator_model.add(
        layers.Conv2D(128, kernel_size=5, strides=(2, 2), padding="same")
    )
    discriminator_model.add(layers.LeakyReLU())
    assert discriminator_model.output_shape == (None, 32, 32, 128)

    discriminator_model.add(
        layers.Conv2D(128, kernel_size=5, strides=(2, 2), padding="same")
    )
    discriminator_model.add(layers.LeakyReLU())
    assert discriminator_model.output_shape == (None, 16, 16, 128)

    discriminator_model.add(layers.Flatten())

    discriminator_model.add(layers.Dense(10))
    discriminator_model.add(layers.Dense(1, activation="sigmoid"))

    return discriminator_model


if __name__ == "__main__":
    discriminator_model = get_discriminator()
    discriminator_model.build()
    tf.keras.utils.plot_model(
        discriminator_model,
        to_file="discriminator.png",
        show_shapes=True,
    )
