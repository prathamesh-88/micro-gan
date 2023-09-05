import os
import time
import tensorflow as tf
from utilities import generate_and_save_images, seed
from constants import global_constants
from lib import (
    get_generator,
    get_discriminator,
    generator_loss,
    discriminator_loss,
    get_normalized_dataSet,
)
from tqdm import tqdm

# Importing models
generator_model = get_generator()
discriminator_model = get_discriminator()

# Defining optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Setting up checkpoints
CHECKPOINT_DIR = "./training_checkpoints"
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator_model=generator_model,
    discriminator_model=discriminator_model,
)

manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)


# Training steps
@tf.function
def train_step(images):
    noise = tf.random.normal(
        [global_constants["BATCH_SIZE"], global_constants["NOISE_DIM"]]
    )
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model(noise, training=True)

        real_output = discriminator_model(images, training=True)
        fake_output = discriminator_model(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_gradient = gen_tape.gradient(
        gen_loss, generator_model.trainable_variables
    )
    discriminator_gradient = disc_tape.gradient(
        disc_loss, discriminator_model.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradient, generator_model.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradient, discriminator_model.trainable_variables)
    )

    return (gen_loss, disc_loss)


# Training loop
def train(dataset, epochs):
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()
    if manager.latest_checkpoint:
        print(f"INFO: {manager.latest_checkpoint} Restored")
    else:
        print("INFO: Starting from scratch")

    gen_losses = []
    disc_losses = []
    gl, dl = 0, 0

    for epoch in tqdm(range(epochs), desc="Training Progress", position=0):
        print(f"Epoch {epoch + 1} started:")
        start = time.time()

        for image_batch in tqdm(
            dataset, desc="Epoch Progress", position=1, leave=False
        ):
            gl, dl = train_step(image_batch)
            gen_losses.append(gl)
            disc_losses.append(dl)

        # Produce images for the GIF as you go
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            generate_and_save_images(generator_model, epoch + 1, seed)
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(
            "Time for epoch {} is {} sec | Generator Loss: {:2f}| Discriminator Loss: {:2f}".format(
                epoch + 1, time.time() - start, gl, dl
            )
        )

    # Generate after the final epoch
    generate_and_save_images(generator_model, epochs + 1, seed)


# Get Dataset
dataset = get_normalized_dataSet()

if __name__ == "__main__":
    train(dataset, global_constants["EPOCHS"])
