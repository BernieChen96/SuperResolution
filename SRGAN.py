from tensorflow.keras import Model, layers, optimizers
import tensorflow as tf
import numpy as np
import getConfig
from glob import glob
import time
from buildSummary import *
import sys
from PIL import Image
import os

config = getConfig.get_config()


class SRGAN(object):

    def __init__(self):
        self.learning_rate = config['learning_rate']
        self.beta_1 = config['beta_1']
        self.lambd = config['lambda']
        self.psnr = config['psnr']
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.adversarial = Adversarial(self.generator, self.discriminator)
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate,
                                         beta_1=self.beta_1, )
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.steps = config['steps']
        self.batch_size = config['batch_size']
        self.img_width = config['image_width']
        self.img_height = config['image_height']
        self.img_channels = config['image_channels']
        self.dataset_train_image_path = config['dataset_train_image_path']
        self.dataset_test_image_path = config['dataset_test_image_path']
        self.dataset_gan_image_path = config['dataset_gan_image_path']
        self.checkpoint_dir = config['checkpoint_dir']
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = config['writer'] + current_time
        self.writer = tf.summary.create_file_writer(log_dir)

    def build_model(self):
        self.generator.build(input_shape=(None, 32, 32, 3))
        self.generator.summary()
        self.discriminator.build(input_shape=(None, 128, 128, 3))
        self.discriminator.summary()
        self.discriminator.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        self.adversarial.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=self.optimizer)

    def train(self):
        data = glob(os.path.join(self.dataset_train_image_path, '*.*'))
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt:
            print("reload pretrained model")
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        train_db = tf.data.Dataset.from_tensor_slices(data)
        train_db = train_db.shuffle(1000).map(self.preprocess).batch(self.batch_size, drop_remainder=True).repeat()
        db_summary(self.writer, "train_db", train_db)
        print('total steps:{}'.format(self.steps))
        for step, imgs in enumerate(train_db):
            start_time = time.time()
            # imgs: [b,256,256,3]
            # g_input: [b,64,64,3]
            g_input = self.down_sample_layer(imgs)
            real_images = imgs
            fake_images = self.generator.predict(g_input)
            real_labels = tf.ones((self.batch_size, int(self.img_width / 2 ** 4), int(self.img_height / 2 ** 4), 1))
            fake_labels = tf.zeros((self.batch_size, int(self.img_width / 2 ** 4), int(self.img_height / 2 ** 4), 1))
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            self.discriminator.trainable = False
            g_loss = self.adversarial.train_on_batch(g_input, [real_labels, real_images])
            loss_summary(self.writer, "g_loss", g_loss[0], step)
            loss_summary(self.writer, "content_loss", g_loss[2], step)
            loss_summary(self.writer, "d_loss", d_loss[0], step)
            loss_summary(self.writer, "psnr", tf.reduce_mean(tf.image.psnr(real_images, fake_images, max_val=2.0)),
                         step)
            if step % 1000 == 0:
                self.test(step=int(step / 1000))
            if (step + 1) % 1000 == 0:
                self.checkpoint.save(file_prefix=checkpoint_prefix)
            sys.stdout.flush()

    def preprocess(self, img):
        # img: dtype: string
        img = tf.io.read_file(img)
        # img: dtype: unit8
        img = tf.image.decode_png(img, channels=self.img_channels)
        # resize会将图像数据类型变为float32型，如果想使用convert_image_dtype就先将其转变为unit8型数据，不想这么做就自己实现归一化
        # img: dtype: float32
        img = tf.image.resize(img, [self.img_width, self.img_height])
        img = tf.cast(img, dtype=tf.uint8)
        # img: dtype:unit8
        # img = tf.image.random_crop(img, [self.img_width, self.img_height, 3])
        # img: dtype: float32 [-1,1]
        img = tf.image.convert_image_dtype(img, dtype=tf.float32) * 2 - 1
        return img

    def test(self, mode="train", step=0):
        if mode == "test":
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt:
                print("reload pretrained model")
                self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        data = glob(os.path.join(self.dataset_test_image_path, '*.*'))
        data = tf.convert_to_tensor(data, dtype=tf.string)
        imgs = np.zeros([len(data), self.img_width, self.img_height, self.img_channels])
        for i in range(len(data)):
            img = self.preprocess(data[i])
            imgs[i] = img
        # imgs: [4,256,256,3]
        # imgs_down: [4,64,64,3]
        imgs_down = self.down_sample_layer(imgs)
        # imgs_sr: [4,256,256,3]
        imgs_sr = self.generator(imgs_down)
        self.save_images(np.asarray(imgs), 2, '{}/gt_{}_hr.png'.format(self.dataset_gan_image_path, step))
        self.save_images(np.asarray(imgs_sr), 2, '{}/test_{}_hr.png'.format(self.dataset_gan_image_path, step))
        self.save_images(np.asarray(imgs_down), 2, '{}/gt_{}_lr.png'.format(self.dataset_gan_image_path, step))

    def save_images(self, val_out, val_block_size, image_path):
        def preprocess(img):
            img = ((img + 1.0) * 127.5).astype(np.uint8)
            # img = img.astype(np.uint8)
            return img

        preprocesed = preprocess(val_out)
        final_image = np.array([])
        single_row = np.array([])
        for b in range(val_out.shape[0]):
            # concat image into a row
            if single_row.size == 0:
                single_row = preprocesed[b, :, :, :]
            else:
                single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

            # concat image row to final_image
            if (b + 1) % val_block_size == 0:
                if final_image.size == 0:
                    final_image = single_row
                else:
                    final_image = np.concatenate((final_image, single_row), axis=0)

                # reset single row
                single_row = np.array([])

        if final_image.shape[2] == 1:
            final_image = np.squeeze(final_image, axis=2)
        Image.fromarray(final_image).save(image_path)


    def down_sample_layer(self, input_x):
        """
        下采样函数
        :param input_x:
        :return:
        """
        K = 4
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            input_x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled


class ResBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(ResBlock, self).__init__()
        self.conv1 = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
        self.relu = layers.ReLU()
        self.norm1 = layers.BatchNormalization(momentum=0.8)
        self.conv2 = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
        self.norm2 = layers.BatchNormalization(momentum=0.8)

    def call(self, inputs, **kwargs):
        outputs = self.norm1(self.relu(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return inputs + outputs


class Generator(Model):
    def __init__(self, res_blocks_num=16, momentum=0.8):
        super(Generator, self).__init__()
        # 初始化超参数
        self.res_blocks_num = res_blocks_num
        self.momentum = momentum
        # 前残差
        self.conv1 = layers.Conv2DTranspose(64, kernel_size=9, strides=1, padding='same', activation='relu')
        # 残差块
        self.res = []
        for i in range(res_blocks_num):
            self.res.append(ResBlock(64, 3, 1))
        # 后残差
        self.conv2 = layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')
        self.norm2 = layers.BatchNormalization(momentum=momentum)

        # 上采样
        self.up_sample1 = layers.UpSampling2D(size=2)
        self.conv3 = layers.Conv2DTranspose(256, kernel_size=3, strides=1, padding='same', activation='relu')
        # 上采样
        self.up_sample2 = layers.UpSampling2D(size=2)
        self.conv4 = layers.Conv2DTranspose(256, kernel_size=3, strides=1, padding='same', activation='relu')
        # 输出卷积层
        self.conv5 = layers.Conv2DTranspose(3, kernel_size=9, strides=1, padding='same', activation='tanh')

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: [b,64,64,3]
        :param training:
        :param mask:
        :return:
        """
        # [b,64,64,64]
        outputs = self.conv1(inputs)
        # 前残差
        skip = outputs
        # [b,64,64,64]
        for i in range(self.res_blocks_num):
            outputs = self.res[i](outputs)
        # [b,64,64,64] 后残差
        outputs = self.norm2(self.conv2(outputs))
        # [b,64,64,64]
        outputs = outputs + skip
        # [b,128,128,64]
        outputs = self.up_sample1(outputs)
        # [b,128,128,256]
        outputs = self.conv3(outputs)
        # [b,256,256,256]
        outputs = self.up_sample2(outputs)
        # [b, 256, 256, 256]
        outputs = self.conv4(outputs)
        # [b, 256, 256, 3]
        outputs = self.conv5(outputs)
        return outputs


class Discriminator(Model):
    def __init__(self, alpha=0.2, momentum=0.8):
        super(Discriminator, self).__init__()
        self.relu = layers.LeakyReLU(alpha=alpha)
        self.conv1 = layers.Conv2D(64, 3, 1, padding='same')
        self.conv2 = layers.Conv2D(64, 3, 2, padding='same')
        self.norm2 = layers.BatchNormalization(momentum=momentum)
        self.conv3 = layers.Conv2D(128, 3, 1, padding='same')
        self.norm3 = layers.BatchNormalization(momentum=momentum)
        self.conv4 = layers.Conv2D(128, 3, 2, padding='same')
        self.norm4 = layers.BatchNormalization(momentum=momentum)
        self.conv5 = layers.Conv2D(256, 3, 1, padding='same')
        self.norm5 = layers.BatchNormalization(momentum=momentum)
        self.conv6 = layers.Conv2D(256, 3, 2, padding='same')
        self.norm6 = layers.BatchNormalization(momentum=momentum)
        self.conv7 = layers.Conv2D(512, 3, 1, padding='same')
        self.norm7 = layers.BatchNormalization(momentum=momentum)
        self.conv8 = layers.Conv2D(512, 3, 2, padding='same')
        self.norm8 = layers.BatchNormalization(momentum=momentum)
        self.fc1 = layers.Dense(1024)
        self.fc2 = layers.Dense(1, activation='sigmoid')

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

        outputs = self.relu(self.fc1(outputs))
        outputs = self.fc2(outputs)
        return outputs


class Adversarial(Model):
    def __init__(self, generator, discriminator):
        super(Adversarial, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:input_low_resolution
        :param training:
        :param mask:
        :return:
        """
        generated_high_resolution_images = self.generator(inputs)
        probs = self.discriminator(generated_high_resolution_images)
        return probs, generated_high_resolution_images
