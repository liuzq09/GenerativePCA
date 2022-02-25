from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA
import tensorflow as tf

import numpy as np
import time
def main(hparams):
    hparams.n_input = np.prod(hparams.image_shape)
    
    maxiter = hparams.max_outer_iter
    utils.print_hparams(hparams)
    
    

    xs_dict = model_input(hparams) # returns the images
    time_elapsed =np.zeros([len(hparams.num_outer_measurement_ls),hparams.num_experiments,len(hparams.noise_std_ls),len(xs_dict.keys()),len(hparams.method_ls)])
    
    print('len xs_dict',len(xs_dict))
    meai=0
    for n_measurement in hparams.num_outer_measurement_ls:

        hparams.num_outer_measurements = n_measurement
        hparams.model_type='dcgan'
        hparams.model_types=['dcgan']
        utils.setup_checkpointing(hparams)  
        # for beta in hparams.beta_ls:
        
        for ri in range(hparams.num_experiments):
            A_outer = utils.get_outer_A(hparams) # Created the random matric A
            for ni in range(len(hparams.noise_std_ls)):
                x_hats_dict = {}
                for hparams.method in hparams.method_ls: 
                    x_hats_dict[hparams.method] = {}
                y_dict = {}
                for key, x in xs_dict.items():
                    print(key)
                    x_batch_dict = {}
                    x_batch_dict[key] = x #placing images in dictionary
                    # if len(x_batch_dict) > hparams.batch_size:
                    #     break
                    x_coll = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()] #Generates the columns of input x
                    x_batch = np.concatenate(x_coll) # Generates entire X                    
                    hparams.noise_std = hparams.noise_std_ls[ni]
                    
                    noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_outer_measurements)
                    if hparams.nonlinear_model == '1bit':
                        y_batch_outer =np.sign(np.matmul(x_batch, A_outer)+noise_batch) # # Multiplication of A and X followed by quantization on 4 levels  /LA.norm(x_batch, axis=(1),keepdims=True)     
                    elif hparams.nonlinear_model == 'cubic':
                        y_batch_outer =(np.matmul(x_batch, A_outer))**3+noise_batch    
                    # for i in range(len(xcidx)):
                    #     x_main_batch[i,:]=V_batch[i,:,xcidx[i]];           
                        
                    # x_main_batch =   x_batch
                    z_opt_batch = np.random.randn(hparams.batch_size, 20) #Input to the generator of the GAN
                    mdi=0
                    for hparams.method in hparams.method_ls:                

                        start = time.time()
                        if hparams.method == 'OneShot':
                            # hparams.num_random_restarts  = 3
                            hparams.max_update_iter=100
                            hparams.model_type='dcgan'
                            hparams.model_types=['dcgan']
                            tf.reset_default_graph()
                            estimators = utils.get_estimators(hparams)
                            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
    
                            
                            x_est_batch = np.matmul(A_outer[np.newaxis,:,:],y_batch_outer[:,:,np.newaxis]).reshape(hparams.batch_size,-1)/hparams.num_outer_measurements
                            
                            estimator = estimators['dcgan']
                            x_est_batch=x_est_batch/LA.norm(x_est_batch, axis=(1),keepdims=True)*LA.norm(x_batch, axis=(1),keepdims=True)
                            # z_opt_batch = np.random.randn(hparams.batch_size, 20) #Input to the generator of the GAN
                            x_hat_batch = estimator(x_est_batch, z_opt_batch, hparams)
                            x_main_batch = x_hat_batch
                            
                            hparams.num_random_restarts  = 1
                        elif hparams.method == 'GD':
                            hparams.model_type='dcgan'
                            hparams.model_types=['dcgan']
                            tf.reset_default_graph()
                            estimators = utils.get_estimators(hparams)
                            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
                            x_main_batch = 0.0 * x_batch
                            hparams.max_update_iter=100
                            for k in range(maxiter):
                 
                                x_est_batch = x_main_batch + hparams.outer_learning_rate/hparams.num_outer_measurements * (np.matmul((y_batch_outer - (np.matmul(x_main_batch, A_outer))), A_outer.T))

                                # estimator = estimators['dcgan']
                                # x_hat_batch = estimator(x_est_batch, z_opt_batch, hparams) # Projectin on the GAN
                                x_hat_batch = x_est_batch
                                x_main_batch = x_hat_batch                             
                            
                        elif hparams.method == 'BIPG':
                            hparams.model_type='dcgan'
                            hparams.model_types=['dcgan']
                            tf.reset_default_graph()
                            estimators = utils.get_estimators(hparams)
                            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
                            x_main_batch = 0.0 * x_batch
                            hparams.max_update_iter=100
                            for k in range(maxiter):
                 
                                x_est_batch = x_main_batch + hparams.outer_learning_rate/hparams.num_outer_measurements * (np.matmul((y_batch_outer - np.sign(np.matmul(x_main_batch, A_outer))), A_outer.T))

                                estimator = estimators['dcgan']
                                x_hat_batch = estimator(x_est_batch, z_opt_batch, hparams) # Projectin on the GAN
                                x_main_batch = x_hat_batch                           
                        elif hparams.method == 'PGD':
                            hparams.max_update_iter=100
                            hparams.model_type='dcgan'
                            hparams.model_types=['dcgan']
                            tf.reset_default_graph()
                            estimators = utils.get_estimators(hparams)
                            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
                            x_main_batch = 0.0 * x_batch
        
                            for k in range(maxiter):
                                x_est_batch = x_main_batch + hparams.outer_learning_rate/hparams.num_outer_measurements * (np.matmul((y_batch_outer - (np.matmul(x_main_batch, A_outer))), A_outer.T))    
                                x_est_batch=x_est_batch/LA.norm(x_est_batch, axis=(1),keepdims=True)*LA.norm(x_batch, axis=(1),keepdims=True)
                                # x_est_batch = x_main_batch + hparams.outer_learning_rate * (np.matmul((y_batch_outer - np.sign(np.matmul(x_main_batch, A_outer))), A_outer.T))
                                #x_est_batch = x_main_batch + hparams.outer_learning_rate * (np.matmul((y_batch_outer - np.matmul(x_main_batch, A_outer)), A_outer.T))
                                # Gradient decent in x is done
                                estimator = estimators['dcgan']
                                x_hat_batch = estimator(x_est_batch, z_opt_batch, hparams) # Projectin on the GAN
                                x_main_batch = x_hat_batch
                        elif hparams.method == 'CSGM':
                            hparams.max_update_iter=100
                            hparams.num_measurements = hparams.num_outer_measurements
                            hparams.model_type='csgm'
                            hparams.model_types=['csgm']
                            tf.reset_default_graph()
                            estimators = utils.get_estimators(hparams)
                            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
                            estimator = estimators[hparams.model_type]
                            x_hat_batch = estimator(A_outer, y_batch_outer, hparams)  
                            x_main_batch = x_hat_batch 
                        elif hparams.method == 'Passive':
                            # x_hats_dict = {hparams.method: {}}
                            gamma = np.sqrt(np.log(hparams.n_input)/hparams.num_outer_measurements)
                            aty = np.matmul(A_outer[np.newaxis,:,:],y_batch_outer[:,:,np.newaxis]).reshape(hparams.batch_size,-1)/hparams.num_outer_measurements
                            norminif=np.amax(aty,axis=1)
                            x_hat_batch = np.zeros_like(aty)
                            x_main_batch = x_hat_batch 
                            for bi in range(0,hparams.batch_size):
                                if norminif[bi] < gamma :
                                    x_hat_batch[bi:bi+1,:] = np.zeros_like(aty[bi:bi+1,:])

                                else:
                                    x_hat_batch[bi:bi+1,:] = np.sign(aty[bi:bi+1,:])*(np.abs(aty[bi:bi+1,:])-gamma)
                                    x_hat_batch[bi:bi+1,:]=x_hat_batch[bi:bi+1,:]/LA.norm(x_hat_batch[bi:bi+1,:], axis=(1),keepdims=True)*LA.norm(x_batch[bi:bi+1,:], axis=(1),keepdims=True)
      
                            x_main_batch = x_hat_batch 
                                    
                        elif hparams.method == 'Lasso-W':
                            hparams.model_type='lasso-wavelet'
                            hparams.model_types=['lasso-wavelet']
                            estimators = utils.get_estimators(hparams)
                            print(estimators)
                            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
                            utils.setup_checkpointing(hparams)
                             
                            estimator = estimators['lasso-wavelet']
                                                   
                            # x_hats_dict = {'lasso': {}}
                          #   x_main_batch = 0.0 * x_batch
                          # #  z_opt_batch = np.random.randn(hparams.batch_size, 20) #Input to the generator of the GAN
                    
                          #   for k in range(maxiter):
                                # x_est_batch = x_main_batch + hparams.outer_learning_rate * (np.matmul((y_batch_outer - (np.matmul(x_main_batch, A_outer))), A_outer.T)) 
                                # if hparams.nonlinear_model == '1bit':
                                #     x_est_batch = x_main_batch + hparams.outer_learning_rate * (np.matmul((y_batch_outer - (np.sign(np.matmul(x_main_batch, A_outer)))), A_outer.T))
                                # elif hparams.nonlinear_model == 'cubic':
                                #     x_est_batch = x_main_batch + hparams.outer_learning_rate * (3/2*(np.matmul((y_batch_outer - ((np.matmul(x_main_batch, A_outer))**3))* np.matmul(x_main_batch, A_outer) **2), A_outer.T))
                                
                            x_hat_batch = estimator(A_outer,y_batch_outer, hparams) # Projectin on the GAN
                            x_main_batch = x_hat_batch
                                
                            
        
                        else :
                            print('The method %s doesn\'t exist, please use PPower, Power, TPower'%hparams.method)                
                        
                        end = time.time()
                        print('Progress: meai,ri,ni,key,mdi,',meai,ri,ni,key,mdi)
                        time_elapsed[meai,ri,ni,key,mdi]=end-start
                        mdi=mdi+1
                        
                        
                        print('x_hat_batch inner',x_hat_batch.shape,x_hat_batch.min(),x_hat_batch.max())
                        
                        # x_hat_batch=x_hat_batch/LA.norm(x_hat_batch, axis=(1),keepdims=True) #x_batch
                        # x_main_batch = x_hat_batch
        
                            
                        # x_main_batch = x_main_batch*LA.norm(x_batch, axis=(1),keepdims=True)
                        # x_main_batch = np.clip(x_main_batch,-1.0,1.0)
                        # x_hat_batch =   x_main_batch  
                        dist = np.linalg.norm(x_batch-x_main_batch)/12288
                        # print('cool')
                        print('recon error',dist)
                
                 
                        for i, key_ in enumerate(x_batch_dict.keys()):
                            x = xs_dict[key]
                            y = y_batch_outer[i]
                            x_hat = x_hat_batch[i]
                
                            # Save the estimate
                            x_hats_dict[hparams.method][key] = x_hat
                            y_dict[key] = y
                            # Compute and store measurement and l2 loss
                            measurement_losses[hparams.model_type][key] = 1.0#utils.get_measurement_loss(x_hat, A_outer, y)
                            l2_losses[hparams.model_type][key] = utils.get_l2_loss(x_hat, x)
                        print('Processed upto image {0} / {1}'.format(key + 1, len(xs_dict)))
                
                        # Checkpointing
                        if (hparams.save_images) and ((key + 1) % hparams.checkpoint_iter == 0):
                            utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, save_image, hparams)
                            #x_hats_dict = {'dcgan' : {}}
                            print('\nProcessed and saved first ', key + 1, 'images\n')
                
                        # x_batch_dict = {}
                    
                        # Final checkpoint
                        if hparams.save_images:
                            utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, save_image, hparams)
                            print('\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict)))
                    
                        if hparams.print_stats:
                            for model_type in hparams.model_types:
                                print(model_type)
                                mean_m_loss = np.mean(measurement_losses[model_type].values())
                                mean_l2_loss = np.mean(l2_losses[model_type].values())
                                print('mean measurement loss = {0}'.format(mean_m_loss))
                                print('mean l2 loss = {0}'.format(mean_l2_loss))
                    
                        if hparams.image_matrix > 0:
                            # hparams.method = 'PPower'
                            outputdir = 'res_sup/nlcsg_%s_%s_sup/'%(hparams.dataset, hparams.nonlinear_model)
                            if not os.path.exists(outputdir):
                                os.mkdir(outputdir)
                            # hparams.savepath=outputdir+'spikedcov_%s_beta_%0.2f_m_%d_r_%d.png'%(hparams.method,beta,hparams.num_outer_measurements,ri)
                            # hparams.savepath=outputdir+'1bit_%s_m_%d_sig_%0.3f_r_%d_k_%d.png'%(hparams.method,hparams.num_outer_measurements,hparams.noise_std,ri,key)
                            
                            # 
                hparams.savepath=outputdir+'%s_%s_m_%d_sig_%0.3f_r_%d.png'%(hparams.dataset, hparams.nonlinear_model, hparams.num_outer_measurements,hparams.noise_std,ri)  
                # print(len(x_hats_dict),x_hats_dict)
                # np.savez(hparams.savepath[:-4]+'.npz',img_gd=xs_dict,img_rec=x_hats_dict)
                # np.savez(hparams.savepath[:-4]+'_mea.npz',img_gd=xs_dict,mea=y_dict,A=A_outer)
                # utils.image_matrix_mls(xs_dict, x_hats_dict, view_image, hparams)
                

                utils.image_matrix_mls(xs_dict, x_hats_dict, view_image, hparams)
                img_rec_ls=[]
                for mi in hparams.method_ls:
                    img_rec_ls = img_rec_ls+[np.stack([vi for vi in x_hats_dict[mi].values()] , axis=0)]
                    
                np.savez(hparams.savepath[:-4]+'.npz',img_gd=np.stack([vi for vi in xs_dict.values()] , axis=0),img_rec=np.stack(img_rec_ls, axis=0))
                np.savez(hparams.savepath[:-4]+'_mea.npz',img_gd=np.stack([vi for vi in xs_dict.values()], axis=0),mea=np.stack([vi for vi in y_dict.values()] , axis=0),A=A_outer)
        meai=meai+1
    np.savez(outputdir+'%s_%s_time_elapsed.npz'%(hparams.dataset, hparams.nonlinear_model),time_elapsed=time_elapsed)

                        # Warn the user that some things were not processsed
                        # if len(x_batch_dict) > 0:
                        #     print('\nDid NOT process last {} images because they did not fill up the last batch.'.format(len(x_batch_dict)))
                        #     print('Consider rerunning lazily with a smaller batch size.')
            
    







if __name__ == '__main__':

    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--pretrained-model-dir', type=str, default='./models/celebA_64_64/', help='Directory containing pretrained model')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--input-type', type=str, default='full-input', help='Where to take input from')
    PARSER.add_argument('--input-path-pattern', type=str, default='./data/celebAtest/*.jpg', help='Pattern to match to get images')
    PARSER.add_argument('--num-input-images', type=int, default=8, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=1, help='How many examples are processed together')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type: as of now supports only gaussian')
    PARSER.add_argument('--noise-std', type=float, default=0.1, help='std dev of noise')
    PARSER.add_argument('--noise-std-ls', metavar='N', type=float, default=[0.1,0.05,0.01], nargs='+', help='std dev of noise')
    PARSER.add_argument('--nonlinear-model', type=str, default='1bit', help='non linear model')
    # Measurement type specific hparams

    PARSER.add_argument('--num-outer-measurement-ls', metavar='N', type=int, default=[300], nargs='+',
                    help='number of measurements') #type=int, default=500, help='number of gaussian measurements(outer)')
    PARSER.add_argument('--beta-ls', metavar='N', type=float, default=[300], nargs='+',
                    help='list of beta') #type=int, default=500, help='number of gaussian measurements(outer)')
    PARSER.add_argument('--method-ls', metavar='N', type=str, default='PPower', nargs='+', help='PPower,Power,TPower')
    
    # Model
    PARSER.add_argument('--num-experiments', type=int, default=10, help='number of experiments')    
    PARSER.add_argument('--model-types', type=str, nargs='+', default=['dcgan'], help='model(s) used for estimation')
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=1.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior_weight', type=float, default=0.001, help='weight on z prior')
    PARSER.add_argument('--dloss1_weight', type=float, default=0.0, help='-log(D(G(z))')
    PARSER.add_argument('--dloss2_weight', type=float, default=0.0, help='log(1-D(G(z))')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='adam', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=100, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=2, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')
    PARSER.add_argument('--outer-learning-rate', type=float, default=1.0, help='learning rate of outer loop GD')
    PARSER.add_argument('--max-outer-iter', type=int, default=30, help='maximum no. of iterations for outer loop GD')
    # LASSO specific hparams
    PARSER.add_argument('--lmbd', type=float, default=0.1, help='lambda : regularization parameter for LASSO')
    PARSER.add_argument('--lasso-solver', type=str, default='sklearn', help='Solver for LASSO')
    # Output
    PARSER.add_argument('--lazy', action='store_true', help='whether the evaluation is lazy')
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--save-stats', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=50, help='checkpoint every x batches')
    PARSER.add_argument('--image-matrix', type=int, default=2,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                        )
    PARSER.add_argument('--gif', action='store_true', help='whether to create a gif')
    PARSER.add_argument('--gif-iter', type=int, default=1, help='save gif frame every x iter')
    PARSER.add_argument('--gif-dir', type=str, default='', help='where to store gif frames')
    HPARAMS = PARSER.parse_args()

    # HPARAMS.image_shape = (28, 28, 1)
    # from mnist_input import model_input
    # from mnist_input import data_input
    # from mnist_utils import view_image, save_image

    if HPARAMS.dataset == 'mnist':
        HPARAMS.image_shape = (28, 28, 1)
        from mnist_input import model_input
        from mnist_utils import view_image, save_image
    elif HPARAMS.dataset == 'celebA':
        HPARAMS.image_shape = (64, 64, 3)
        from celebA_input import model_input
        from celebA_utils import view_image, save_image
    else:
        raise NotImplementedError
    main(HPARAMS)
