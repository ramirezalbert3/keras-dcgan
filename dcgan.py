import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


def construct_discriminator(image_shape: tuple, conv_conf: dict):
    """ The discriminator tries to classify images as real or fake """

    discriminator = Sequential()
    discriminator.add(Conv2D(filters=64, input_shape=image_shape, name='input', **conv_conf))
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
                        input_shape=(1, 1, noise_size),
                        name='input'))
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


def generate_noise(batch_size: int, noise_size: int):
    """ Return a noisy array with shape [batch_size, 1, 1, noise_size] """
    # TODO: figure out who should be generating this noise (is it DCGAN because of noise size?)
    return np.random.normal(0, 1, size=(batch_size,) + (1, 1, noise_size))

class DCGAN:
    def __init__(self, img_shape: tuple, noise_size: int = 100,
                 discriminator: Sequential = None, generator: Sequential = None,
                 conv_conf: dict = None):

        if conv_conf is None:
            conv_conf = {'kernel_size': (5, 5),
                         'strides': (2, 2),
                         'padding': 'same',
                         'data_format': 'channels_last',
                         'kernel_initializer': 'glorot_uniform'}

        self.generator = construct_generator(img_shape, noise_size, conv_conf)
        self.discriminator = construct_discriminator(img_shape, conv_conf)

        self.gan = Sequential()
        # Only false for the adversarial model
        self.discriminator.trainable = False
        self.gan.add(self.generator)
        self.gan.add(self.discriminator)

        optimizer = Adam(lr=0.00015, beta_1=0.5)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer,
                         metrics=None)

    def generate(self, noise: np.array):
        """ Generate images based on noise data """
        return self.generator.predict(noise)

    def train_discriminator(self, real_images: np.array, generated_images: np.array):
        """ Train discriminator by feeding real data with y=aprox(1) and fake data with y=0 """
        assert(len(real_images) == len(generated_images))  # assume first axis is batch size
        batch_size = len(real_images)
        # as advised by Goodfellow in NIPS2016, we label real data with around 0.9 and fake with 0
        real_y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.1
        generated_y = np.zeros(batch_size)
        self.discriminator.trainable = True
        d_loss = self.discriminator.train_on_batch(real_images, real_y)
        d_loss += self.discriminator.train_on_batch(generated_images, generated_y)
        self.discriminator.trainable = False
        return d_loss

    def train_generator(self, noise: np.array):
        """ Train generator by misleading the discriminator, feeding fake data with y=aprox(1) """
        batch_size = len(noise)
        generated_y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.1
        return self.gan.train_on_batch(noise, generated_y)

    def save_model(self, path: str = None):
        if path is None:
            path = 'models/'
        self.discriminator.trainable = True  # TODO: ???
        self.discriminator.save(path + 'discriminator.hdf5')
        self.generator.save(path + 'generator.hdf5')

    @staticmethod
    def load_model(discriminator_path: str = None, generator_path: str = None):
        if discriminator_path is None:
            discriminator_path = 'models/discriminator.hdf5'
        if generator_path is None:
            generator_path = 'models/generator.hdf5'
        discriminator = load_model(discriminator_path)
        generator = load_model(generator_path)
        _, img_x, img_y, channels = discriminator.get_layer("input").input_shape  # drop batch_size axis
        img_shape = (img_x, img_y, channels)
        noise_size = generator.get_layer("input").input_shape[3]
        # TODO: log all this
        return DCGAN(img_shape, noise_size, discriminator, generator)
