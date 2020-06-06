from SRGAN import *
import getConfig
import os

config = getConfig.get_config()


def main():
    srgan = SRGAN()
    mode = config['mode']
    if mode == 'train':
        srgan.build_model()
        srgan.train()
    elif mode == 'test':
        srgan.test(mode=mode)


if __name__ == '__main__':
    main()
