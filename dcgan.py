import time
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_dataset(batch_size):
    dataset_generator = ImageDataGenerator()
    (x_train, y_train), _ = mnist.load_data()
    # Reduce training data, keep only X batches per epoch
    batches = 50
    x_train = x_train[:batch_size*batches, :, :]
    y_train = y_train[:batch_size*batches]

    new_shape = x_train.shape + (1,)
    x_train = x_train.reshape(new_shape)
    dataset_generator = dataset_generator.flow(
        x_train, y_train,
        batch_size=batch_size)

    return dataset_generator, len(x_train)


# Creates the discriminator model. This model tries to classify images as real
# or fake.
def construct_discriminator(image_shape):

    discriminator = Sequential()
    discriminator.add(Conv2D(filters=64, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform',
                             input_shape=(image_shape)))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=128, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=256, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=512, kernel_size=(5, 5),
                             strides=(2, 2), padding='same',
                             data_format='channels_last',
                             kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))

    optimizer = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=None)

    return discriminator


# Creates the generator model. This model has an input of random noise and
# generates an image that will try mislead the discriminator.
def construct_generator():

    generator = Sequential()

    generator.add(Dense(units=7 * 7 * 512,
                        kernel_initializer='glorot_uniform',
                        input_shape=(1, 1, 100)))
    generator.add(Reshape(target_shape=(7, 7, 512)))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=128, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=1, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(Activation('tanh'))

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    generator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=None)

    return generator


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
def train_dcgan(batch_size, epochs, image_shape):
    # Build the adversarial model that consists in the generator output
    # connected to the discriminator
    # generator = construct_generator()
    # discriminator = construct_discriminator(image_shape)
    generator = load_model('models/generator.hdf5')
    discriminator = load_model('models/discriminator.hdf5')

    gan = Sequential()
    # Only false for the adversarial model
    discriminator.trainable = False
    gan.add(generator)
    gan.add(discriminator)

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer,
                metrics=None)

    # Create a dataset Generator with help of keras
    dataset_generator, dataset_size = load_dataset(batch_size)

    # 11788 is the total number of images on the bird dataset
    number_of_batches = int(dataset_size / batch_size)

    # Variables that will be used to plot the losses from the discriminator and
    # the adversarial models
    generator_loss = np.empty(shape=1)
    discriminator_loss = np.empty(shape=1)
    batches = np.empty(shape=1)

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
            real_images, _ = dataset_generator.next()

            # The last batch is smaller than the other ones, so we need to
            # take that into account
            current_batch_size = real_images.shape[0]

            # Generate noise
            noise = np.random.normal(0, 1,
                                     size=(current_batch_size,) + (1, 1, 100))

            # Generate images
            generated_images = generator.predict(noise)

            # Add some noise to the labels that will be
            # fed to the discriminator
            # TODO: not advised by goofellow to not label fake date with y=0
            real_y = (np.ones(current_batch_size) -
                      np.random.random_sample(current_batch_size) * 0.2)
            fake_y = np.random.random_sample(current_batch_size) * 0.2

            # Let's train the discriminator
            discriminator.trainable = True

            d_loss = discriminator.train_on_batch(real_images, real_y)
            d_loss += discriminator.train_on_batch(generated_images, fake_y)

            discriminator_loss = np.append(discriminator_loss, d_loss)

            # Now it's time to train the generator
            discriminator.trainable = False

            noise = np.random.normal(0, 1,
                                     size=(current_batch_size * 2,) +
                                     (1, 1, 100))

            # We try to mislead the discriminator by giving the opposite labels
            # TODO: not advised by goofellow to not label fake date with y=0
            fake_y = (np.ones(current_batch_size * 2) -
                      np.random.random_sample(current_batch_size * 2) * 0.2)

            g_loss = gan.train_on_batch(noise, fake_y)
            generator_loss = np.append(generator_loss, g_loss)
            batches = np.append(batches, current_batch)

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
        discriminator.trainable = True  # TODO: ???
        generator.save('models/generator.hdf5')
        discriminator.save('models/discriminator.hdf5')

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
    epochs = 10
    train_dcgan(batch_size, epochs, image_shape)


if __name__ == "__main__":
    main()
