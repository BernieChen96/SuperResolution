from SRGAN import *
import tensorflow as tf
import getConfig

config = getConfig.get_config()


def main():
    srgan = SRGAN()
    mode = config['mode']
    if mode == 'train':
        srgan.build_model()
        srgan.train()
    elif mode == 'test':
        srgan.test()


if __name__ == '__main__':
    main()
