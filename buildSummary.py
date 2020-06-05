import datetime
import tensorflow as tf
from matplotlib import pyplot as plt
import io
from matplotlib import patches
import numpy as np


def model_summary(writer):
    tf.summary.g
    pass


def loss_summary(writer, title, scalar, step):
    with writer.as_default():
        tf.summary.scalar(title, scalar, step)
        writer.flush()


def db_summary(writer, title, db):
    sample = next(iter(db))
    build_summary_image(writer, title, sample, step=0)


def build_summary_image(writer, title, images, step):
    figure = image_grid(np.asarray(images))
    with writer.as_default():
        tf.summary.image(title, plot_to_image(figure), step=step)


def image_grid(images):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    images = ((images + 1.0) * 127.5).astype(np.uint8)
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
