import datetime
import tensorflow as tf
from matplotlib import pyplot as plt
import io
from matplotlib import patches
import numpy as np


def db_summary(title, db):
    sample = next(iter(db))
    build_summary_image(title, sample, step=0)


def scalar_summary(title, scalar, step):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    with summary_writer.as_default():
        tf.summary.scalar(title, scalar, step=step)


def build_summary_image(title, images, step):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    figure = image_grid(images)
    with summary_writer.as_default():
        tf.summary.image(title, plot_to_image(figure), step=step)


def image_grid(images):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))

    for i in range(4):
        # Start next subplot.
        plt.subplot(2, 2, i + 1, title='name')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)

    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def train_db_visualize(yolo, db):
    # imgs:[b,512,512,3]
    # imgs_boxes:[b,40,5]
    imgs, imgs_boxes = next(iter(db))
    img, img_boxes = imgs[1], imgs_boxes[1]
    f, ax1 = plt.subplots(figsize=(10, 10))
    # display the JPEGImages, [512,512,3]
    ax1.imshow(img)
    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(yolo.num_classes, 3), dtype='uint8')
    for x1, y1, x2, y2, l in img_boxes:  # [40,5]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        w = x2 - x1
        h = y2 - y1

        if l == 0:
            break
        else:  # ignore invalid boxes
            l = int((l - 1).numpy())
            color = np.array([int(c) for c in COLORS[l]]) / 255.
            rect = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
    plt.show()
