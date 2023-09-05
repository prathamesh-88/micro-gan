import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img

# from tensorflow.keras import layers


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    plt.figure(figsize=(10, 10))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        img = array_to_img(predictions[i] * 127.5 + 127.5)
        plt.imshow(img)
        plt.axis("off")

    plt.savefig("./results/image_at_epoch_{:04d}.png".format(epoch))
    plt.close()
