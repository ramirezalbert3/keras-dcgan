import time
import matplotlib
matplotlib.use('TkAgg')  # MacOS hack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist

from dcgan import DCGAN, generate_noise


def load_dataset(batch_size, max_batches):
    dataset_generator = ImageDataGenerator()
    (x_train, y_train), _ = mnist.load_data()
    # Reduce training data, keep only X batches per epoch
    if len(x_train) > batch_size * max_batches:
        x_train = x_train[:batch_size*max_batches, :, :]
        y_train = y_train[:batch_size*max_batches]

    new_shape = x_train.shape + (1,)
    x_train = x_train.reshape(new_shape)
    dataset_generator = dataset_generator.flow(x_train, y_train,
                                               batch_size=batch_size)

    return dataset_generator, len(x_train)


# Displays a figure of the generated images and saves them in as .png image
def save_generated_images(generated_images, epoch, batch_number):
    plt.figure(figsize=(8, 8), num=2)
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace=0, hspace=0)

    for i in range(64):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = np.squeeze(generated_images[i, :, :, :])  # drop the dim when only 1-channel (28, 28, 1) to (28, 28)
        fig = plt.imshow(image.astype(np.float), cmap='gray')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    save_name = 'generated_images/generatedSamples_epoch' + str(
        epoch + 1) + '_batch' + str(batch_number + 1) + '.png'

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.pause(3)
    plt.show()


# Main train function
def train_dcgan(batch_size, epochs, image_shape, max_batches):
    noise_size = 100
    gan = DCGAN(image_shape, noise_size)
    # gan = DCGAN.load_model()
    # Create a dataset Generator with help of keras
    dataset_generator, dataset_size = load_dataset(batch_size, max_batches)

    # 11788 is the total number of images on the bird dataset
    number_of_batches = int(dataset_size / batch_size)

    generator_loss = []
    discriminator_loss = []
    batches = []

    # Allow plot updates inside for loop
    plt.ion()

    current_batch = 0

    # Let's train the DCGAN for n epochs
    batches_trained = 0
    for epoch in range(epochs):

        print("Epoch {:3d}/{}:".format(epoch+1, epochs))

        for batch_number in range(number_of_batches):
            batches_trained += 1
            start_time = time.time()

            # Get the current batch and normalize the images between -1 and 1
            # TODO: class that handles data (loading, batching, normalizing, etc)
            real_images, _ = dataset_generator.next()

            # The last batch is smaller than the other ones, so we need to
            # take that into account
            current_batch_size = real_images.shape[0]

            # Generate images
            noise = generate_noise(current_batch_size, noise_size)
            generated_images = gan.generate(noise)

            d_loss = gan.train_discriminator(real_images, generated_images)
            discriminator_loss.append(d_loss)

            noise = generate_noise(2*current_batch_size, noise_size)
            g_loss = gan.train_generator(noise)
            generator_loss.append(g_loss)

            batches.append(current_batch)

            # Every 50 batches_trained show and save images
            if(batches_trained % 50 == 0 and
               current_batch_size == batch_size):
                save_generated_images(generated_images, epoch, batch_number)

            time_elapsed = time.time() - start_time

            # Display and plot the results
            print('\tBatch {:3d}/{}: generator loss | discriminator loss: {:5.2f} | {:5.2f} in {:5.2f}s'.format(
                batch_number + 1, number_of_batches, g_loss, d_loss, time_elapsed))

            current_batch += 1

        # Save model every epoch
        gan.save_model()

        # Each epoch update the loss graphs
        plt.figure(1)
        plt.plot(batches, generator_loss, color='green',
                 label='Generator Loss')
        plt.plot(batches, discriminator_loss, color='blue',
                 label='Discriminator Loss')
        plt.title("DCGAN Train")
        plt.xlabel("Batch Iteration")
        plt.ylabel("Loss")
        if epoch == 0:
            plt.legend()
        plt.pause(0.0000000001)
        plt.show()
        plt.savefig('trainingLossPlot.png')


def main():
    batch_size = 64
    image_shape = (28, 28, 1)
    epochs = 20
    max_batches = 50
    train_dcgan(batch_size, epochs, image_shape, max_batches)


if __name__ == "__main__":
    main()
