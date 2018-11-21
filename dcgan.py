from keras.models import Sequential  # TODO: try to use functional model
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

# main reference is [4]. https://github.com/Goldesel23/DCGAN-for-Bird-Generation
# because it uses tips/tricks on top of baselines and because its nicely explained


class DCGAN:
    def __init__(self):
        pass
