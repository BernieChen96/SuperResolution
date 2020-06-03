from tensorflow.keras import Model, layers, Sequential, optimizers
import tensorflow as tf
import pysnooper
import numpy as np
import getConfig
from glob import glob
import os
import time
from buildSummary import *
import sys

config = getConfig.get_config()


class SRGAN(object):

    def __init__(self):
        self.learning_rate = config['learning_rate']
        self.beta_1 = config['beta_1']
        self.beta_2 = config['beta_2']
        self.lambd = config['lambda']
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.optimizer_g = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                    beta_1=self.beta_1,
                                                    beta_2=self.beta_2)
        self.optimizer_d = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                    beta_1=self.beta_1,
                                                    beta_2=self.beta_2)
        self.optimizer_content = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                          beta_1=self.beta_1,
                                                          beta_2=self.beta_2)
        self.checkpoint = tf.train.Checkpoint(optimizer_g=self.optimizer_g,
                                              optimizer_d=self.optimizer_d,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.steps = config['steps']
        self.batch_size = config['batch_size']
        self.img_width = config['image_width']
        self.img_height = config['image_height']
        self.img_channels = config['image_channels']
        self.dataset_train_image_path = config['dataset_train_image_path']
        self.checkpoint_dir = config['checkpoint_dir']

    def build_model(self):
        self.generator.build(input_shape=(1, 64, 64, 3))
        self.generator.summary()
        self.discriminator.build(input_shape=(1, 256, 256, 3))
        self.discriminator.summary()
        # 使用Checkpoint，保存训练模型
        return self.generator, self.discriminator

    def mse_content_loss(self, real, fake):
        return tf.reduce_mean(tf.square(real - fake))

    def inference_loss(self, real, fake):
        def inference_mse_content_loss(real, fake):
            return tf.reduce_mean(tf.square(real - fake))

        def inference_adversarial_loss(x, y, w=1, type_='gan'):
            if type_ == 'gan':
                return w * tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            elif type_ == 'lsgan':
                return w * (x - y) ** 2
            else:
                raise ValueError('no {} loss type'.format(type_))

        content_loss = inference_mse_content_loss(real, fake)
        d_real_logits = self.discriminator(real)
        d_fake_logits = self.discriminator(fake)
        d_fake_loss = tf.reduce_mean(inference_adversarial_loss(d_real_logits, tf.ones_like(d_real_logits)))
        d_real_loss = tf.reduce_mean(inference_adversarial_loss(d_fake_logits, tf.zeros_like(d_fake_logits)))
        g_fake_loss = tf.reduce_mean(inference_adversarial_loss(d_fake_logits, tf.ones_like(d_fake_logits)))

        d_loss = self.lambd * (d_fake_loss + d_real_loss)
        g_loss = content_loss + self.lambd * g_fake_loss

        return d_loss, g_loss, content_loss

    def train(self):
        data = glob(os.path.join(self.dataset_train_image_path, '*.*'))
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt:
            print("reload pretrained model")
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        train_db = tf.data.Dataset.from_tensor_slices(data)
        train_db = train_db.shuffle(1000).map(self.preprocess).batch(self.batch_size).repeat()
        db_summary("train_db", train_db)
        print('total steps:{}'.format(self.steps))

        for step, imgs in enumerate(train_db):
            start_time = time.time()
            # imgs: [b,256,256,3]
            # g_input: [b,64,64,3]
            g_input = self.down_sample_layer(imgs)
            real = imgs
            # with tf.GradientTape() as tape:
            #     fake = self.generator(g_input)
            #     content_loss = self.mse_content_loss(real, fake)
            #     psnr = tf.image.psnr(real, fake, max_val=1.0)
            # grads = tape.gradient(content_loss, self.generator.trainable_variables)
            # self.optimizer_g.apply_gradients(zip(grads, self.generator.trainable_variables))
            # print("step:", step, 'content_loss:', float(content_loss), "psnr:", psnr)
            with tf.GradientTape() as tape:
                fake = self.generator(g_input)
                _, g_loss, _ = self.inference_loss(real, fake)
                psnr = tf.image.psnr(real, fake, max_val=1.0)
            grads = tape.gradient(g_loss, self.generator.trainable_variables)
            self.optimizer_g.apply_gradients(zip(grads, self.generator.trainable_variables))
            print("step:", step, 'g_loss:', float(g_loss), "psnr:", float(tf.reduce_mean(psnr)))

            with tf.GradientTape() as tape:
                fake = self.generator(g_input)
                d_loss, _, _ = self.inference_loss(real, fake)
            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.optimizer_d.apply_gradients(zip(grads, self.discriminator.trainable_variables))
            print("step:", step, 'd_loss:', float(d_loss))
            end_time = time.time()
            scalar_summary("g_loss", g_loss, step)
            scalar_summary("psnr", float(tf.reduce_mean(psnr)), step)
            scalar_summary("d_loss", d_loss, step)
            print(step,
                  "Estimated time remaining {} seconds".format(int((self.steps - step) * (end_time - start_time))))
            if (step + 1) % 10 == 0:
                self.checkpoint.save(file_prefix=checkpoint_prefix)
            if step > self.steps:
                break
            sys.stdout.flush()

    def preprocess(self, img):
        # img: dtype: string
        img = tf.io.read_file(img)
        # img: dtype: unit8
        img = tf.image.decode_png(img, channels=self.img_channels)
        # resize会将图像数据类型变为float32型，如果想使用convert_image_dtype就先将其转变为unit8型数据，不想这么做就自己实现归一化
        # img: dtype: float32
        img = tf.image.resize(img, [2 * self.img_width, 2 * self.img_height])
        img = tf.cast(img, dtype=tf.uint8)
        # img: dtype:unit8
        img = tf.image.random_crop(img, [self.img_width, self.img_height, 3])
        # img: dtype: float32
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        return img

    def down_sample_layer(self, input_x):
        K = 4
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            input_x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.res_block1 = self.res_block(64, 3, 1)
        self.res_block2 = self.res_block(64, 3, 1)
        self.res_block3 = self.res_block(64, 3, 1)
        self.res_block4 = self.res_block(64, 3, 1)
        self.res_block5 = self.res_block(64, 3, 1)

        self.conv2 = layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')
        self.norm2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(256, kernel_size=3, strides=1, padding='same')

        self.conv4 = layers.Conv2DTranspose(256, kernel_size=3, strides=1, padding='same')

        self.conv5 = layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same')

    def res_block(self, filters, kernel_size, strides):
        res_block = Sequential()
        res_block.add(layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same'))
        res_block.add(layers.BatchNormalization())
        res_block.add(layers.LeakyReLU(alpha=0))
        res_block.add(layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same'))
        res_block.add(layers.BatchNormalization())
        return res_block

    def pixel_shuffle_layer(self, x, r, n_split):
        """

        :param x: [b,256,256,256]
        :param r: 2
        :param n_split:64
        :return:
        """

        def PS(x, r):
            """

            :param x: [b,256,256,4]
            :param r: 2
            :return:
            """
            # bs:batchsz, a:256, b:256, c:256
            bs, a, b, c = x.get_shape().as_list()
            # [b,256,256,2,2]
            x = tf.reshape(x, (bs, a, b, r, r))
            # [b,256,256,2,2]
            x = tf.transpose(x, [0, 1, 2, 4, 3])
            # list: [b,1,256,2,2],...
            x = tf.split(x, a, 1)
            # squeeze(x[1]): [b,256,2,2], x: [b,256,512,2]
            x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)
            # list: [b,1,512,2]
            x = tf.split(x, b, 1)
            # squeeze(x[1]): [b,512,2], x: [b,512,512]
            x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)
            # [b,512,512,1]
            return tf.reshape(x, (bs, a * r, b * r, 1))

        # xc:list [b,256,256,4],[b,256,256,4]...
        xc = tf.split(x, n_split, 3)
        # PS: [b,512,512,1]
        return tf.concat([PS(x_, r) for x_ in xc], 3)

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: [b,64,64,3]
        :param training:
        :param mask:
        :return:
        """
        # [b,64,64,64]
        outputs = self.conv1(inputs)
        skip = outputs
        # [b,64,64,64]
        outputs = outputs + self.res_block1(outputs)
        # [b,64,64,64]
        outputs = outputs + self.res_block2(outputs)
        # [b,64,64,64]
        outputs = outputs + self.res_block3(outputs)
        # [b,64,64,64]
        outputs = outputs + self.res_block4(outputs)
        # [b,64,64,64]
        outputs = outputs + self.res_block5(outputs)
        # [b,64,64,64]
        outputs = self.norm2(self.conv2(outputs))
        # [b,64,64,64]
        outputs = outputs + skip
        # [b,64,64,256]
        outputs = self.conv3(outputs)
        # [b,128,128,64]
        outputs = tf.nn.relu(self.pixel_shuffle_layer(outputs, 2, 64))
        # [b,128,128,256]
        outputs = self.conv4(outputs)
        # (1, 256, 256, 64)
        outputs = tf.nn.relu(self.pixel_shuffle_layer(outputs, 2, 64))
        # (1, 256, 256, 3)
        outputs = self.conv5(outputs)
        outputs = tf.nn.tanh(outputs)
        return outputs


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.relu = layers.LeakyReLU(alpha=0.2)
        self.conv1 = layers.Conv2D(64, 3, 1, padding='same')
        self.conv2 = layers.Conv2D(64, 3, 2, padding='same')
        self.norm2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(128, 3, 1, padding='same')
        self.norm3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(128, 3, 2, padding='same')
        self.norm4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(256, 3, 1, padding='same')
        self.norm5 = layers.BatchNormalization()
        self.conv6 = layers.Conv2D(256, 3, 2, padding='same')
        self.norm6 = layers.BatchNormalization()
        self.conv7 = layers.Conv2D(512, 3, 1, padding='same')
        self.norm7 = layers.BatchNormalization()
        self.conv8 = layers.Conv2D(512, 3, 2, padding='same')
        self.norm8 = layers.BatchNormalization()
        self.flat = layers.Flatten()
        self.fc1 = layers.Dense(1024)
        self.fc2 = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs:[b,256,256,3]
        :param training:
        :param mask:
        :return:
        """
        # [1,256,256,64]
        outputs = self.relu(self.conv1(inputs))
        # [1, 128, 128, 64]
        outputs = self.relu(self.norm2(self.conv2(outputs)))
        # [1, 128, 128, 128]
        outputs = self.relu(self.norm3(self.conv3(outputs)))
        # [1,64,64,128]
        outputs = self.relu(self.norm4(self.conv4(outputs)))
        # [1,64,64,256]
        outputs = self.relu(self.norm5(self.conv5(outputs)))
        # [1,32,32,256]
        outputs = self.relu(self.norm6(self.conv6(outputs)))
        # [1,32,32,512]
        outputs = self.relu(self.norm7(self.conv7(outputs)))
        # [1,16,16,512]
        outputs = self.relu(self.norm8(self.conv8(outputs)))
        # [1,131072]
        outputs = self.flat(outputs)

        outputs = self.relu(self.fc1(outputs))
        outputs = self.fc2(outputs)
        return outputs


if __name__ == '__main__':
    srgan = SRGAN()
    srgan.train()
