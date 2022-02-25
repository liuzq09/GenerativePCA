# This file based on : https://jmetzen.github.io/notebooks/vae.ipynb
# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, R0902


import tensorflow as tf
import utils
from ops import *

# class Hparams(object):
#     def __init__(self):
#         self.n_hidden_recog_1 = 500  # 1st layer encoder neurons
#         self.n_hidden_recog_2 = 500  # 2nd layer encoder neurons
#         self.n_hidden_gener_1 = 500  # 1st layer decoder neurons
#         self.n_hidden_gener_2 = 500  # 2nd layer decoder neurons
#         self.n_input = 784           # MNIST data input (img shape: 28*28)
#         self.n_z = 20                # dimensionality of latent space
#         self.transfer_fct = tf.nn.softplus

class Hparams(object):
    def __init__(self):
        self.n_input = 784           # MNIST data input (img shape: 28*28)
        self.n_z = 62               # dimensionality of latent space
        self.transfer_fct = tf.nn.softplus

def encoder(hparams, x_ph, scope_name, reuse):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.get_variable('w1', initializer=utils.xavier_init(hparams.n_input, hparams.n_hidden_recog_1))
        b1 = tf.get_variable('b1', initializer=tf.zeros([hparams.n_hidden_recog_1], dtype=tf.float32))
        hidden1 = hparams.transfer_fct(tf.matmul(x_ph, w1) + b1)

        w2 = tf.get_variable('w2', initializer=utils.xavier_init(hparams.n_hidden_recog_1, hparams.n_hidden_recog_2))
        b2 = tf.get_variable('b2', initializer=tf.zeros([hparams.n_hidden_recog_2], dtype=tf.float32))
        hidden2 = hparams.transfer_fct(tf.matmul(hidden1, w2) + b2)

        w3 = tf.get_variable('w3', initializer=utils.xavier_init(hparams.n_hidden_recog_2, hparams.n_z))
        b3 = tf.get_variable('b3', initializer=tf.zeros([hparams.n_z], dtype=tf.float32))
        z_mean = tf.matmul(hidden2, w3) + b3

        w4 = tf.get_variable('w4', initializer=utils.xavier_init(hparams.n_hidden_recog_2, hparams.n_z))
        b4 = tf.get_variable('b4', initializer=tf.zeros([hparams.n_z], dtype=tf.float32))
        z_log_sigma_sq = tf.matmul(hidden2, w4) + b4

    return z_mean, z_log_sigma_sq

def generator(hparams, z, scope_name='', reuse=False):
    batch_size = z.get_shape().as_list()[0]
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope("generator", reuse=reuse):
        net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=False, scope='g_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=False, scope='g_bn2'))
        net = tf.reshape(net, [batch_size, 7, 7, 128])
        net = tf.nn.relu(
            bn(deconv2d(net, [batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=False,
                scope='g_bn3'))

        out = tf.nn.sigmoid(deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))
        out = tf.reshape(out, [batch_size, 784])
        return out

def get_loss(x, logits, z_mean, z_log_sigma_sq):
    reconstr_losses = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), 1)
    latent_losses = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    total_loss = tf.reduce_mean(reconstr_losses + latent_losses, name='total_loss')
    return total_loss


def get_z_var(hparams, batch_size):
    z = tf.Variable(tf.random_normal((batch_size, 62)), name='z')
    return z


def gen_restore_vars():
    restore_vars = ['generator/g_fc1/Matrix', 'generator/g_fc1/bias', 'generator/g_bn1/beta', 
                'generator/g_bn1/gamma', 'generator/g_bn1/moving_mean', 'generator/g_bn1/moving_variance',
                'generator/g_fc2/Matrix', 'generator/g_fc2/bias', 'generator/g_bn2/beta', 'generator/g_bn2/gamma', 
                'generator/g_bn2/moving_mean', 'generator/g_bn2/moving_variance', 'generator/g_dc3/w', 'generator/g_dc3/biases',
                'generator/g_bn3/beta', 'generator/g_bn3/gamma', 'generator/g_bn3/moving_mean', 
                 'generator/g_bn3/moving_variance', 'generator/g_dc4/w', 'generator/g_dc4/biases'
                ]
    return restore_vars
