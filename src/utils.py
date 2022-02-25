# Files of this project is modified versions of 'https://github.com/AshishBora/csgm', which
#comes with the MIT licence: https://github.com/AshishBora/csgm/blob/master/LICENSE

import os
import pickle
import shutil
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import Lasso
from numpy import linalg as LA
from scipy.linalg import dft
import itertools
from itertools import combinations
from scipy.linalg import hadamard

import mnist_estimators
import fashion_estimators
import celebA_estimators

from sklearn.linear_model import Lasso
from l1regls import l1regls
from cvxopt import matrix


class BestKeeper(object):
    """Class to keep the best stuff"""
    def __init__(self, hparams):
        self.batch_size = hparams.batch_size
        self.losses_val_best = [1e10 for _ in range(hparams.batch_size)]
        self.x_hat_batch_val_best = np.zeros((hparams.batch_size, hparams.n_input))
        self.z_batch_val_best = np.zeros((hparams.batch_size, 20))

    def report(self, x_hat_batch_val, z_batch_val, losses_val):
        print(self.batch_size)
        for i in range(self.batch_size):
            if losses_val[i] < self.losses_val_best[i]:
                self.x_hat_batch_val_best[i, :] = x_hat_batch_val[i, :]
                self.z_batch_val_best[i, :] = z_batch_val[i, :]
                self.losses_val_best[i] = losses_val[i]

    def get_best(self):
        return self.x_hat_batch_val_best,self.z_batch_val_best
class BestKeeper_fashion(object):
    """Class to keep the best stuff"""
    def __init__(self, hparams):
        self.batch_size = hparams.batch_size
        self.losses_val_best = [1e10 for _ in range(hparams.batch_size)]
        self.x_hat_batch_val_best = np.zeros((hparams.batch_size, hparams.n_input))
        self.z_batch_val_best = np.zeros((hparams.batch_size, 62))

    def report(self, x_hat_batch_val, z_batch_val, losses_val):
        print(self.batch_size)
        for i in range(self.batch_size):
            if losses_val[i] < self.losses_val_best[i]:
                self.x_hat_batch_val_best[i, :] = x_hat_batch_val[i, :]
                self.z_batch_val_best[i, :] = z_batch_val[i, :]
                self.losses_val_best[i] = losses_val[i]

    def get_best(self):
        return self.x_hat_batch_val_best,self.z_batch_val_best  
    
class BestKeeperCSGM(object):
    """Class to keep the best stuff"""
    def __init__(self, hparams):
        self.batch_size = hparams.batch_size
        self.losses_val_best = [1e10 for _ in range(hparams.batch_size)]
        self.x_hat_batch_val_best = np.zeros((hparams.batch_size, hparams.n_input))

    def report(self, x_hat_batch_val, losses_val):
        for i in range(self.batch_size):
            if losses_val[i] < self.losses_val_best[i]:
                self.x_hat_batch_val_best[i, :] = x_hat_batch_val[i, :]
                self.losses_val_best[i] = losses_val[i]

    def get_best(self):
        return self.x_hat_batch_val_best    
    
class BestKeeper_dcgn(object):
    """Class to keep the best stuff"""
    def __init__(self, hparams):
        self.batch_size = hparams.batch_size
        self.losses_val_best = [1e10 for _ in range(hparams.batch_size)]
        self.x_hat_batch_val_best = np.zeros((hparams.batch_size, hparams.n_input))

    def report(self, x_hat_batch_val, losses_val):
        for i in range(self.batch_size):
            if losses_val[i] < self.losses_val_best[i]:
                self.x_hat_batch_val_best[i, :] = x_hat_batch_val[i, :]
                self.losses_val_best[i] = losses_val[i]

    def get_best(self):
        return self.x_hat_batch_val_best

def get_l2_loss(image1, image2):
    """Get L2 loss between the two images"""
    assert image1.shape == image2.shape
    return np.mean((image1 - image2)**2)

def get_rel_error(image1, image2):
    return (np.linalg.norm(image1-image2)/(np.linalg.norm(image1)))

def get_ssim(image1, image2):
    return ssim(image1,image2)

def get_measurement_loss(x_hat, A, y):
    """Get measurement loss of the estimated image"""
    if A is None:
        y_hat = x_hat
    else:
        y_hat = np.matmul(x_hat, A)
    assert y_hat.shape == y.shape
    print("get_measurement_loss")
    return np.mean((y - y_hat) ** 2)



def save_to_pickle(data, pkl_filepath):
    """Save the data to a pickle file"""
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data


def get_estimator(hparams, model_type):
    if hparams.dataset == 'mnist':
        if model_type == 'vae':
            estimator = mnist_estimators.vae_estimator(hparams)
        elif model_type == 'lasso':
            estimator = mnist_estimators.lasso_estimator(hparams)
        elif model_type == 'omp':
            estimator = mnist_estimators.omp_estimator(hparams)
        elif model_type == 'learned':
            estimator = mnist_estimators.learned_estimator(hparams)
        else:
            raise NotImplementedError
    if hparams.dataset == 'fashion':
        if model_type == 'vae':
            estimator = fashion_estimators.vae_estimator(hparams)
        elif model_type == 'lasso':
            estimator = fashion_estimators.lasso_estimator(hparams)
        elif model_type == 'omp':
            estimator = fashion_estimators.omp_estimator(hparams)
        elif model_type == 'learned':
            estimator = fashion_estimators.learned_estimator(hparams)
        else:
            raise NotImplementedError           
    elif hparams.dataset == 'celebA':
        if model_type == 'lasso-dct':
            estimator = celebA_estimators.lasso_dct_estimator(hparams)
        elif model_type == 'csgm':
            estimator = celebA_estimators.csgm_estimator(hparams)            
        elif model_type == 'lasso-wavelet':
            estimator = celebA_estimators.lasso_wavelet_estimator(hparams)
        elif model_type == 'lasso-wavelet-ycbcr':
            estimator = celebA_estimators.lasso_wavelet_ycbcr_estimator(hparams)
        elif model_type == 'k-sparse-wavelet':
            estimator = celebA_estimators.k_sparse_wavelet_estimator(hparams)
        elif model_type == 'dcgan':
            estimator = celebA_estimators.dcgan_estimator(hparams)
        else:
            raise NotImplementedError            

    return estimator


def get_estimators(hparams):
    estimators = {model_type: get_estimator(hparams, model_type) for model_type in hparams.model_types}
    return estimators


def setup_checkpointing(hparams):
    # Set up checkpoint directories
    for model_type in hparams.model_types:
        print(hparams.model_types)
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        set_up_dir(checkpoint_dir)



def save_images(est_images, save_image, hparams):
    """Save a batch of images to png files"""
    for model_type in hparams.model_types:
        for image_num, image in est_images[model_type].iteritems():
            save_path = get_save_paths(hparams, image_num)[model_type]
            image = image.reshape(hparams.image_shape)
            save_image(image, save_path)


def checkpoint(est_images, measurement_losses, l2_losses, save_image, hparams):
    """Save images, measurement losses and L2 losses for a batch"""
    if hparams.save_images:
        save_images(est_images, save_image, hparams)

    if hparams.save_stats:
        for model_type in hparams.model_types:
            m_losses_filepath, l2_losses_filepath = get_pkl_filepaths(hparams, model_type)
            save_to_pickle(measurement_losses[model_type], m_losses_filepath)
            save_to_pickle(l2_losses[model_type], l2_losses_filepath)


def load_checkpoints(hparams):
    measurement_losses, l2_losses = {}, {}
    if hparams.save_images:
        # Load pickled loss dictionaries
        for model_type in hparams.model_types:
            m_losses_filepath, l2_losses_filepath = get_pkl_filepaths(hparams, model_type)
            measurement_losses[model_type] = load_if_pickled(m_losses_filepath)
            l2_losses[model_type] = load_if_pickled(l2_losses_filepath)
    else:
        for model_type in hparams.model_types:
            measurement_losses[model_type] = {}
            l2_losses[model_type] = {}
    return measurement_losses, l2_losses


def image_matrix(images, est_images, view_image, hparams, alg_labels=True):
    """Display images"""


    figure_height = 1 + len(hparams.model_types)

    # fig = plt.figure(figsize=[2*len(images), 2*figure_height])
    fig, axes = plt.subplots(nrows=figure_height, ncols=1, figsize=[2*len(images), 2*figure_height])
    outer_counter = 0
    inner_counter = 0


    # Show original images
    # outer_counter += 1
    # print(images.values())
    for image in images.values():
        inner_counter += 1
        # ax = fig.add_subplot(figure_height, 1, outer_counter)
        # axes[outer_counter].plot()
        axes[outer_counter].get_xaxis().set_visible(False)
        axes[outer_counter].get_yaxis().set_ticks([])
        axes[outer_counter].spines["top"].set_visible(False)
        axes[outer_counter].spines["bottom"].set_visible(False)            
        axes[outer_counter].spines["right"].set_visible(False)
        axes[outer_counter].spines["left"].set_visible(False)        
        if alg_labels:
            # axes[outer_counter].set_ylabel('Original', fontsize=14)
            axes[outer_counter].yaxis.set_label_text('Original',fontsize=12)
            
        _ = fig.add_subplot(figure_height, len(images), inner_counter)
        view_image(image, hparams)


    for model_type in hparams.model_types:
        outer_counter += 1
        for image in est_images[model_type].values():

            inner_counter += 1
            # ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            # axes[outer_counter].plot()
            axes[outer_counter].get_xaxis().set_visible(False)
            axes[outer_counter].get_yaxis().set_ticks([])
            axes[outer_counter].spines["top"].set_visible(False)
            axes[outer_counter].spines["bottom"].set_visible(False)            
            axes[outer_counter].spines["right"].set_visible(False)
            axes[outer_counter].spines["left"].set_visible(False)

            if alg_labels:
                # ax.set_ylabel(model_type, fontsize=14)
                # ax.set_ylabel('GPCA',fontsize=12)
                axes[outer_counter].yaxis.set_label_text(hparams.method,fontsize=12)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image, hparams)

    if hparams.image_matrix >= 2:
       # save_path = get_matrix_save_path(hparams)
        if hasattr(hparams,"savepath"):
            plt.savefig(hparams.savepath)
        else:
            plt.savefig('rec_fig.png')

    if hparams.image_matrix in [1, 3]:
        plt.show()

def image_matrix_mls(images, est_images, view_image, hparams, alg_labels=True):
    """Display images"""


    figure_height = 1 + len(hparams.method_ls)

    # fig = plt.figure(figsize=[2*len(images), 2*figure_height])
    fig, axes = plt.subplots(nrows=figure_height, ncols=1, figsize=[2*len(images), 2*figure_height])
    outer_counter = 0
    inner_counter = 0


    # Show original images
    # outer_counter += 1
    # print(images.values())
    for image in images.values():
        inner_counter += 1
        # ax = fig.add_subplot(figure_height, 1, outer_counter)
        # axes[outer_counter].plot()
        axes[outer_counter].get_xaxis().set_visible(False)
        axes[outer_counter].get_yaxis().set_ticks([])
        axes[outer_counter].spines["top"].set_visible(False)
        axes[outer_counter].spines["bottom"].set_visible(False)            
        axes[outer_counter].spines["right"].set_visible(False)
        axes[outer_counter].spines["left"].set_visible(False)        
        if alg_labels:
            # axes[outer_counter].set_ylabel('Original', fontsize=14)
            axes[outer_counter].yaxis.set_label_text('Original',fontsize=12)
            
        _ = fig.add_subplot(figure_height, len(images), inner_counter)
        view_image(image, hparams)


    for model_type in hparams.method_ls:
        outer_counter += 1
        for image in est_images[model_type].values():

            inner_counter += 1
            # ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            # axes[outer_counter].plot()
            axes[outer_counter].get_xaxis().set_visible(False)
            axes[outer_counter].get_yaxis().set_ticks([])
            axes[outer_counter].spines["top"].set_visible(False)
            axes[outer_counter].spines["bottom"].set_visible(False)            
            axes[outer_counter].spines["right"].set_visible(False)
            axes[outer_counter].spines["left"].set_visible(False)

            if alg_labels:
                # ax.set_ylabel(model_type, fontsize=14)
                # ax.set_ylabel('GPCA',fontsize=12)
                axes[outer_counter].yaxis.set_label_text(model_type,fontsize=12)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image, hparams)

    if hparams.image_matrix >= 2:
       # save_path = get_matrix_save_path(hparams)
        if hasattr(hparams,"savepath"):
            plt.savefig(hparams.savepath,bbox_inches='tight')
        else:
            plt.savefig('rec_fig.png',bbox_inches='tight')

    if hparams.image_matrix in [1, 3]:
        plt.show()
        
def image_plot(imgdict, view_image, hparams, alg_labels=True,font_size=24):
    """Display images"""


    figure_height = len(imgdict)

    # fig = plt.figure(figsize=[2*len(images), 2*figure_height])
    fig, axes = plt.subplots(nrows=figure_height, ncols=1, figsize=[2*len(imgdict['Original']), 2*figure_height])
    outer_counter = 0
    inner_counter = 0


    # Show original images
    # outer_counter += 1
    # print(images.values())
    # for image in images.values():
    #     inner_counter += 1
    #     # ax = fig.add_subplot(figure_height, 1, outer_counter)
    #     # axes[outer_counter].plot()
    #     axes[outer_counter].get_xaxis().set_visible(False)
    #     axes[outer_counter].get_yaxis().set_ticks([])
    #     axes[outer_counter].spines["top"].set_visible(False)
    #     axes[outer_counter].spines["bottom"].set_visible(False)            
    #     axes[outer_counter].spines["right"].set_visible(False)
    #     axes[outer_counter].spines["left"].set_visible(False)        
    #     if alg_labels:
    #         # axes[outer_counter].set_ylabel('Original', fontsize=14)
    #         axes[outer_counter].yaxis.set_label_text('Original',fontsize=12)
            
    #     _ = fig.add_subplot(figure_height, len(images), inner_counter)
    #     view_image(image, hparams)


    for model_type in hparams.model_types:
        
        for image in imgdict[model_type]:

            inner_counter += 1
            # ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            # axes[outer_counter].plot()
            axes[outer_counter].get_xaxis().set_visible(False)
            axes[outer_counter].get_yaxis().set_ticks([])
            axes[outer_counter].spines["top"].set_visible(False)
            axes[outer_counter].spines["bottom"].set_visible(False)            
            axes[outer_counter].spines["right"].set_visible(False)
            axes[outer_counter].spines["left"].set_visible(False)

            if alg_labels:
                # ax.set_ylabel(model_type, fontsize=14)
                # ax.set_ylabel('GPCA',fontsize=12)
                if model_type == 'OneShot-FP':
                    axes[outer_counter].yaxis.set_label_text(model_type,fontsize=18)
                else:
                    axes[outer_counter].yaxis.set_label_text(model_type,fontsize=font_size)
            _ = fig.add_subplot(figure_height, len(imgdict['Original']), inner_counter)
            view_image(image, hparams)
            # plt.tight_layout()
        outer_counter += 1


    if hasattr(hparams,"savepath"):
        plt.savefig(hparams.savepath,bbox_inches='tight')
    else:
        plt.savefig('rec_fig.png')

    if hparams.image_matrix in [1, 3]:
        plt.show()

def plot_image(image, cmap=None):
    """Show the image"""
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame = frame.imshow(image, cmap=cmap)


def get_checkpoint_dir(hparams, model_type):
    base_dir = './estimated/{0}/{1}/{2}/{3}/'.format(
        hparams.input_type,
        hparams.measurement_type,
        hparams.num_outer_measurements,
        model_type
        )
    if model_type in ['vae']:
        dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            hparams.mloss1_weight,
            hparams.mloss2_weight,
            hparams.zprior_weight,
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.decay_lr,
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    elif model_type in ['lasso', 'lasso-dct', 'lasso-wavelet','lasso-wavelet-ycbcr']:
        dir_name = '{}'.format(
            hparams.lmbd,
        )
      
    elif model_type in ['dcgan']:
        dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            hparams.mloss1_weight,
            hparams.mloss2_weight,
            hparams.zprior_weight,
            hparams.dloss1_weight,
            hparams.dloss2_weight,
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.decay_lr,
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    else:
        raise NotImplementedError

    ckpt_dir = base_dir + dir_name + '/'

    return ckpt_dir


def get_pkl_filepaths(hparams, model_type):
    """Return paths for the pickle files"""
    checkpoint_dir = get_checkpoint_dir(hparams, model_type)
    m_losses_filepath = checkpoint_dir + 'measurement_losses.pkl'
    l2_losses_filepath = checkpoint_dir + 'l2_losses.pkl'
    return m_losses_filepath, l2_losses_filepath


def get_save_paths(hparams, image_num):
    save_paths = {}
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        save_paths[model_type] = checkpoint_dir + '{0}.png'.format(image_num)
    return save_paths


def get_matrix_save_path(hparams):
    save_path = './estimated/{0}/{1}/{2}/matrix_{3}.png'.format(
        hparams.input_type,
        hparams.measurement_type,
        '_'.join(hparams.model_types)
    )
    return save_path


def set_up_dir(directory, clean=False):
    if os.path.exists(directory):
        if clean:
            shutil.rmtree(directory)
    else:
        os.makedirs(directory)


def print_hparams(hparams):
    # print ''
    for temp in dir(hparams):
        if temp[:1] != '_':
            print('{0} = {1}'.format(temp, getattr(hparams, temp)))
    print('')


def get_learning_rate(global_step, hparams):
    if hparams.decay_lr:
        return tf.train.exponential_decay(hparams.learning_rate,
                                          global_step,
                                          50,
                                          0.7,
                                          staircase=True)
    else:
        return tf.constant(hparams.learning_rate)


def get_optimizer(learning_rate, hparams):
    if hparams.optimizer_type == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    if hparams.optimizer_type == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, hparams.momentum)
    elif hparams.optimizer_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate)
    elif hparams.optimizer_type == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif hparams.optimizer_type == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif hparams.optimizer_type == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate)
    else:
        raise Exception('Optimizer ' + hparams.optimizer_type + ' not supported')

def get_outer_A(hparams):
    A_outer =  np.random.randn(hparams.n_input,hparams.num_outer_measurements)
    # A_outer =  (1.0 / np.sqrt(hparams.num_outer_measurements)) * A_outer
    return A_outer
        
   # lists=np.random.randint(low=0, high=hparams.n_input-1, size=100)
  #  lists=lists.sort()
  #  A_outer=A[lists]
   #  A_outer=A.T
  #   A_outer =  (1.0 / (np.sqrt(hparams.num_outer_measurements))) * A_outer
  # #  A= np.random.randn(hparams.n_input,hparams.num_outer_measurements)

   

  #   return A_outer

def get_spikedcov_A(hparams):
    # A_outer =  np.eye(hparams.n_input,hparams.num_outer_measurements)
    # dimmin = min(hparams.n_input,hparams.num_outer_measurements)
    # A_outer[range(dimmin),range(dimmin)]= np.random.randn(dimmin)
    A_outer =  np.random.randn(1,1,hparams.num_outer_measurements)
    # A_outer =  (1.0 / np.sqrt(hparams.num_outer_measurements)) * A_outer
    return A_outer

def get_checkpoint_path(ckpt_dir):
    ckpt_dir = os.path.abspath(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = os.path.join(ckpt_dir,
                                 ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')
        ckpt_path = ''
    return ckpt_path


def save_plot(is_save, save_path):
    if is_save:
        pdf = PdfPages(save_path)
        pdf.savefig(bbox_inches='tight')
        pdf.close()


def solve_lasso(A_val, y_val, hparams):
    if hparams.lasso_solver == 'sklearn':
        lasso_est = Lasso(alpha=hparams.lmbd)
        lasso_est.fit(A_val.T, y_val)
        x_hat = lasso_est.coef_
        x_hat = np.reshape(x_hat, [-1])
    if hparams.lasso_solver == 'cvxopt':
        A_mat = matrix(A_val.T)
        y_mat = matrix(y_val)
        x_hat_mat = l1regls(A_mat, y_mat)
        x_hat = np.asarray(x_hat_mat)
        x_hat = np.reshape(x_hat, [-1])
    return x_hat


def get_opt_reinit_op(opt, var_list, global_step):
    opt_slots = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in var_list]
    # if isinstance(opt, tf.train.AdamOptimizer):
    #     opt_slots.extend([opt._beta1_power, opt._beta2_power])
    all_opt_variables = opt_slots + var_list + [global_step]
    opt_reinit_op = tf.variables_initializer(all_opt_variables)
    return opt_reinit_op

