from tensorflow.keras import Model,layers
import tensorflow as tf

class SRGAN(object):

    def __init__(self):
        pass


class Generator(Model):
    def __init__(self):
        self.conv1=layers.Conv2DTranspose()
        pass

    def call(self, inputs, training=None, mask=None):
        pass


class Discriminator(Model):
    def __init__(self):
        pass

    def call(self, inputs, training=None, mask=None):
        pass
