import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


def construct_discriminator(image_shape: tuple, conv_filters: list, conv_conf: dict):
    """ The discriminator tries to classify images as real or fake """

    discriminator = Sequential()
    for idx, f in enumerate(conv_filters):
        if idx == 0:
            discriminator.add(Conv2D(filters=f, input_shape=image_shape,
                                     name='input', **conv_conf))
        else:
            discriminator.add(Conv2D(filters=f, **conv_conf))
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


def construct_generator(img_shape: tuple, noise_size: int, conv_filters: list, conv_conf: dict):
    """"" The generator gets noise of size noise_size and outputs an image of img_shape """""
    assert (img_shape[0] == img_shape[1])
    assert (len(img_shape) == 3)  # assert W, H, Channels
    base = img_shape[0]
    used_filters = []
    # TODO: less hard-coding for conv_conf, its a lot better now however
    #       Every deconvolution (at least with current hardcoded stride/padding) doubles size

    # Find how many convolutions we can apply, and where do we start from
    while base % 2 == 0:
        used_filters.append(conv_filters.pop())
        base = base // 2
        if len(conv_filters) == 0:
            break

    generator = Sequential()

    for idx, f in enumerate(used_filters):
        if idx == 0:
            generator.add(Dense(units=base * base * f,
                                kernel_initializer='glorot_uniform',
                                input_shape=(1, 1, noise_size),
                                name='input'))
            generator.add(Reshape(target_shape=(base, base, f)))
        else:
            generator.add(Conv2DTranspose(filters=f, **conv_conf))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=img_shape[2], **conv_conf))
    generator.add(Activation('tanh'))

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    generator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=None)

    _, x, y, channels = generator.output_shape  # first is batch_size = None
    assert ((x, y, channels) == img_shape)

    return generator


def generate_noise(batch_size: int, noise_size: int):
    """ Return a noisy array with shape [batch_size, 1, 1, noise_size] """
    # TODO: figure out who should be generating this noise (is it DCGAN because of noise size?)
    return np.random.normal(0, 1, size=(batch_size,) + (1, 1, noise_size))

class DCGAN:
    def __init__(self, img_shape: tuple, noise_size: int = 100,
                 discriminator: Sequential = None, generator: Sequential = None,
                 conv_filters: list = None, conv_conf: dict = None):

        if conv_conf is None:
            conv_conf = {'kernel_size': (5, 5),
                         'strides': (2, 2),
                         'padding': 'same',
                         'data_format': 'channels_last',
                         'kernel_initializer': 'glorot_uniform'}
        else:
            print('Use at own risk, probably not ready')

        if conv_filters is None:
            conv_filters = [64, 128, 256, 512]
        else:
            print('Use at own risk, probably not ready')

        if generator is None:
            self._generator = construct_generator(img_shape, noise_size, conv_filters, conv_conf)
        else:
            self._generator = generator

        if discriminator is None:
            self._discriminator = construct_discriminator(img_shape, conv_filters, conv_conf)
        else:
            self._discriminator = discriminator

        self._gan = Sequential()
        self._discriminator.trainable = False
        self._gan.add(self._generator)
        self._gan.add(self._discriminator)

        optimizer = Adam(lr=0.00015, beta_1=0.5)
        self._gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=None)

    def generate(self, noise: np.array):
        """ Generate images based on noise data """
        return self._generator.predict(noise)

    def train_discriminator(self, real_images: np.array, generated_images: np.array):
        """ Train discriminator by feeding real data with y=aprox(1) and fake data with y=0 """
        # TODO: Goodfellow in NIPS2016 advises we label real data with around 0.9 and fake with 0
        #       but got better results with this approach
        real_y = np.ones(len(real_images)) - np.random.random_sample(len(real_images)) * 0.2
        generated_y = np.random.random_sample(len(generated_images)) * 0.2
        self._discriminator.trainable = True
        d_loss = self._discriminator.train_on_batch(real_images, real_y)
        d_loss += self._discriminator.train_on_batch(generated_images, generated_y)
        self._discriminator.trainable = False
        return d_loss

    def train_generator(self, noise: np.array):
        """ Train generator by misleading the discriminator, feeding fake data with y=aprox(1) """
        batch_size = len(noise)
        generated_y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
        return self._gan.train_on_batch(noise, generated_y)

    def save_model(self, path: str = None):
        if path is None:
            path = 'models/'
        self._discriminator.trainable = True  # TODO: found bug mentions related to saving and the trainable flag
        self._discriminator.save(path + 'discriminator.hdf5')
        self._generator.save(path + 'generator.hdf5')

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
