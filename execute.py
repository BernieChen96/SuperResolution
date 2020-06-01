from SRGAN import *
import tensorflow as tf

# 是否需要执行的步骤
is_crop = True
is_testing = True
is_training = False


def main():
    srgan = SRGAN(dataset_dir='data/',
                  is_crop=is_crop,
                  batch_size=8,
                  input_height=128, input_width=128, input_channels=3)
    srgan.build_model()
    if is_training:
        srgan.train()
    if is_testing:
        srgan.test()


if __name__ == '__main__':
    main()
