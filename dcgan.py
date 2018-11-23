from keras.models import Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


def construct_discriminator(image_shape: tuple, conv_conf: dict):
    """ The discriminator tries to classify images as real or fake """

    discriminator = Sequential()
    discriminator.add(Conv2D(filters=64, input_shape=image_shape, **conv_conf))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=128, **conv_conf))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=256, **conv_conf))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=512, **conv_conf))
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


def construct_generator(img_shape: tuple, noise_size: int, conv_conf: dict):
    """"" The generator gets noise of size noise_size and outputs an image of img_shape """""

    generator = Sequential()

    generator.add(Dense(units=7 * 7 * 512,
                        kernel_initializer='glorot_uniform',
                        input_shape=(1, 1, noise_size)))
    generator.add(Reshape(target_shape=(7, 7, 512)))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=128, **conv_conf))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=1, **conv_conf))
    generator.add(Activation('tanh'))

    # TODO: assert(result_shape == img_shape)

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    generator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=None)

    return generator


class DCGAN:
    def __init__(self, img_shape: tuple, noise_size: int = 100, conv_conf: dict = None):
        if conv_conf is None:
            conv_conf = {'kernel_size': (5, 5),
                         'strides': (2, 2),
                         'padding': 'same',
                         'data_format': 'channels_last',
                         'kernel_initializer': 'glorot_uniform'}
        self.generator = construct_generator(img_shape, noise_size, conv_conf)
        self.discriminator = construct_discriminator(img_shape, conv_conf)
        # generator = load_model('models/generator.hdf5')
        # discriminator = load_model('models/discriminator.hdf5')

        self.gan = Sequential()
        # Only false for the adversarial model
        self.discriminator.trainable = False
        self.gan.add(self.generator)
        self.gan.add(self.discriminator)

        optimizer = Adam(lr=0.00015, beta_1=0.5)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer,
                         metrics=None)
