"""Inputs for MNIST dataset"""

import math
import numpy as np
#import mnist_model_def
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

NUM_TEST_IMAGES = 10000


def get_random_test_subset(mnist, sample_size):
    """Get a small random subset of test images"""
    idxs = np.random.choice(NUM_TEST_IMAGES, sample_size)
    images = [mnist.test.images[idx] for idx in idxs]
    images = {i: image for (i, image) in enumerate(images)}
    return images


def sample_generator_images(hparams):
    """Sample random images from the generator"""

    # Create the generator
    _, x_hat, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)

    # Get a session
    sess = tf.Session()

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    images = {}
    counter = 0
    rounds = int(math.ceil(hparams.num_input_images/hparams.batch_size))
    for _ in range(rounds):
        images_mat = sess.run(x_hat)
        for (_, image) in enumerate(images_mat):
            if counter < hparams.num_input_images:
                images[counter] = image
                counter += 1

    # Reset TensorFlow graph
    sess.close()
    tf.reset_default_graph()

    return images


def model_input(hparams):
    """Create input tensors"""

    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

    if hparams.input_type == 'full-input':
        images = {i: image for (i, image) in enumerate(mnist.test.images[:hparams.num_input_images])}
    elif hparams.input_type == 'random-test':
        images = get_random_test_subset(mnist, hparams.num_input_images)
    elif hparams.input_type == 'gen-span':
        images = sample_generator_images(hparams)
    else:
        raise NotImplementedError

    return images

def load_mnist(dataset_name):
    import os, gzip
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def model_input_fashion(hparams):
    """Create input tensors"""
    fashion_mnist, _ = load_mnist('fashion-mnist')
    print('fashion_mnist', fashion_mnist.shape)

    if hparams.input_type == 'full-input':
        # images = {i: image for (i, image) in }
        images = {i: image for (i, image) in enumerate(fashion_mnist[20:(20+hparams.num_input_images)])}
    elif hparams.input_type == 'random-test':
        images = get_random_test_subset(mnist, hparams.num_input_images)
    elif hparams.input_type == 'gen-span':
        images = sample_generator_images(hparams)
    else:
        raise NotImplementedError

    return images

def data_input(hparams):
    """Create input tensors"""

    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

    if hparams.input_type == 'full-input':
        images = {i: image for (i, image) in enumerate(mnist.test.images[:400])}
    elif hparams.input_type == 'random-test':
        images = get_random_test_subset(mnist, hparams.num_input_images)
    elif hparams.input_type == 'gen-span':
        images = sample_generator_images(hparams)
    else:
        raise NotImplementedError

    return images
