from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA
from celebA_estimators import wavelet_basis

import numpy as np

def main(hparams):
    hparams.n_input = np.prod(hparams.image_shape)
    hparams.model_type='dcgan'
    maxiter = hparams.max_outer_iter
    utils.print_hparams(hparams)
    
    
    
    measurement_losses, l2_losses = utils.load_checkpoints(hparams)

    
    x_batch_dict = {}
    estimators = utils.get_estimators(hparams)
    for n_measurement in hparams.num_outer_measurement_ls:
        hparams.num_outer_measurements = n_measurement
        
        xs_dict = model_input(hparams) # returns the images
        utils.setup_checkpointing(hparams)
        # for beta in hparams.beta_ls:

            
        for ri in range(0,hparams.num_experiments):
            x_hats_dict = {}
            for hparams.method in hparams.method_ls: 
                x_hats_dict[hparams.method] = {}
            for key, x in xs_dict.items():
                print(key)
                x_batch_dict = {}
                x_batch_dict[key] = x #placing images in dictionary
            # if len(x_batch_dict) > hparams.batch_size:
            #     break
                x_coll = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()] #Generates the columns of input x
                x_batch = (np.concatenate(x_coll)+1.0)/2.0 
                # print('x_batch',x_batch.shape,x_batch.min(),x_batch.max())            
                
                A_outer = utils.get_outer_A(hparams) # Created the random matric A
            #    noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, 100)
                y_batch_outer =np.abs(np.matmul(x_batch, A_outer)) # Multiplication of A and X followed by quantization on 4 levels  /LA.norm(x_batch, axis=(1),keepdims=True)
                lwb = 1.0
                upb = 5.0
                lamb = y_batch_outer.mean(axis=1, keepdims=False)*np.sqrt(np.pi/2.0)
                if 0:                
       
                    V_batch = np.zeros((x_batch.shape[0],x_batch.shape[1],x_batch.shape[1]),dtype=np.float32)
                    for si in range(y_batch_outer.shape[1]):
                        aat=np.matmul(A_outer[:,si:si+1],A_outer[:,si:si+1].transpose((1, 0)))#-np.eye(x_batch.shape[1], dtype=np.float32)
                        for bi in range(y_batch_outer.shape[0]):               
                                if y_batch_outer[bi,si]>lamb[bi]*lwb  and y_batch_outer[bi,si]<lamb[bi]*upb:                     
                                    V_batch[bi,:,:]=V_batch[bi,:,:]+(y_batch_outer[bi,si])*aat
                    V_batch = V_batch/y_batch_outer.shape[1]                     
                   
                    xcidx=np.diagonal(V_batch, axis1=1, axis2=2).argmax(1)  
                    
                for hparams.method in hparams.method_ls: 
                    x_main_batch = np.matmul((y_batch_outer),A_outer.transpose((1,0)))
                    if 0:
                        x_main_batch = 0.0 * x_batch        
                        for i in range(len(xcidx)):
                            x_main_batch[i,:]=V_batch[i,:,xcidx[i]];           
                        
                    # x_main_batch =   x_batch
                    z_opt_batch = np.random.randn(hparams.batch_size, 20) #Input to the generator of the GAN
            
                    for k in range(maxiter):
                        
                        
            
                        if 0:
                            x_est_batch = np.matmul(V_batch,x_main_batch.reshape((x_main_batch.shape[0],x_main_batch.shape[1],1))).reshape((x_main_batch.shape[0],x_main_batch.shape[1]))/LA.norm(x_main_batch, axis=(1),keepdims=True)
                        
                        else:
                            
                            # print('y shape:',y_batch_outer.shape)
                            Vx_batch =  np.zeros((x_batch.shape[0],x_batch.shape[1]),dtype=np.float32)
                            for si in range(y_batch_outer.shape[1]):
                                for bi in range(y_batch_outer.shape[0]):   
              
                                    if y_batch_outer[bi,si]>lamb[bi]*lwb  and y_batch_outer[bi,si]<lamb[bi]*upb:
                                        # print(np.matmul(A_outer[:,si:si+1].transpose((1, 0)), x_main_batch[bi:bi+1,:].transpose((1, 0))),A_outer[:,si].shape,Vx_batch[bi,:].shape)
                                        Vx_batch[bi,:]=Vx_batch[bi,:]+y_batch_outer[bi,si]*A_outer[:,si]*(np.matmul(A_outer[:,si:si+1].transpose((1, 0)), x_main_batch[bi:bi+1,:].transpose((1, 0))).reshape((1)))
                            Vx_batch = Vx_batch/y_batch_outer.shape[1]

                            
                            x_est_batch = Vx_batch/LA.norm(Vx_batch, axis=(1),keepdims=True)                        
                        
                        # if hparams.method == 'PPower':
                        #     estimator = estimators['dcgan']
                        #     x_est_batch=x_est_batch/LA.norm(x_est_batch, axis=(1),keepdims=True)*LA.norm(x_batch, axis=(1),keepdims=True)
                        #     print('x_batch inner',x_batch.shape,x_batch.min(),x_batch.max())
                        #     x_hat_batch = estimator(x_est_batch, hparams) #z_opt_batch
                            
                        # elif hparams.method == 'Power':
                        #     x_hat_batch = x_est_batch
                        # elif hparams.method == 'TPower':
                        #     nokbiggest = 150
                        #     argsx=np.argsort(np.absolute(x_est_batch),1)
                        #     x_hat_batch = np.zeros_like(x_est_batch)
                        #     x_hat_batch[np.repeat(np.arange(x_est_batch.shape[0])[:,np.newaxis],nokbiggest,1),argsx[:,-nokbiggest:]] = x_est_batch[np.repeat(np.arange(x_est_batch.shape[0])[:,np.newaxis],nokbiggest,1),argsx[:,-nokbiggest:]]
                        # else:
                        #     print('The method %s doesn\'t exist, please use PPower, Power, TPower'%hparams.method)
                                # x_est_batch = x_batch
                        if hparams.method == 'PPower':
                            estimator = estimators['dcgan']
                            x_est_batch=x_est_batch/LA.norm(x_est_batch, axis=(1),keepdims=True)*LA.norm(x_batch, axis=(1),keepdims=True)
                            print('x_batch inner',x_batch.shape,x_batch.min(),x_batch.max())
                            x_hat_batch = (estimator(x_est_batch*2.0-1.0,z_opt_batch, hparams)+1.0)/2.0 #z_opt_batch
                            
                        elif hparams.method == 'Power':
                            x_hat_batch = x_est_batch
                            
                        elif hparams.method == 'TPower':
                            nokbiggest = 2000
                            argsx=np.argsort(np.absolute(x_est_batch),1)
                            x_hat_batch = np.zeros_like(x_est_batch)
                            x_hat_batch[np.repeat(np.arange(x_est_batch.shape[0])[:,np.newaxis],nokbiggest,1),argsx[:,-nokbiggest:]] = x_est_batch[np.repeat(np.arange(x_est_batch.shape[0])[:,np.newaxis],nokbiggest,1),argsx[:,-nokbiggest:]]
                        elif hparams.method == 'TPowerW':
                            W = wavelet_basis()
#                            print('W shape',W.shape,'x shape:',x_est_batch.shape)
                            xw_est_batch= np.matmul(W[np.newaxis,:,:], x_est_batch[:,:,np.newaxis]).reshape((-1,W.shape[0]))
                            nokbiggest = 2000
                            argsx=np.argsort(np.absolute(xw_est_batch),1)                            
                            xw_hat_batch = np.zeros_like(xw_est_batch)
                            # print(np.repeat(np.arange(x_est_batch.shape[0])[:,np.newaxis],nokbiggest,1))
                            xw_hat_batch[np.repeat(np.arange(x_est_batch.shape[0])[:,np.newaxis],nokbiggest,1),argsx[:,-nokbiggest:]] = xw_est_batch[np.repeat(np.arange(xw_est_batch.shape[0])[:,np.newaxis],nokbiggest,1),argsx[:,-nokbiggest:]]    
                            x_hat_batch= np.matmul(W[np.newaxis,:,:].transpose((0,2,1)), xw_hat_batch[:,:,np.newaxis]).reshape((-1,W.shape[1]))
                        else:
                            print('The method %s doesn\'t exist, please use PPower, Power, TPower'%hparams.method)                                
                                
                        x_hat_batch=x_hat_batch/LA.norm(x_hat_batch, axis=(1),keepdims=True) #x_batch
                        x_main_batch = x_hat_batch
                        
                    
        
            
                    x_main_batch = x_main_batch*LA.norm(x_batch, axis=(1),keepdims=True)* 2.0 - 1.0
                    x_hat_batch = x_main_batch
                    
                    # x_main_batch = np.clip(x_main_batch,-1.0,1.0)
                    dist = np.linalg.norm(x_batch-x_main_batch)/12288
                    # print('cool')
                    print('recon error',dist)                
                    for i, key in enumerate(x_batch_dict.keys()):
                        x = xs_dict[key]
                        y = y_batch_outer[i]
                        x_hat = x_hat_batch[i]
            
                        # Save the estimate
                        x_hats_dict[hparams.method][key] = x_hat
            
                        # Compute and store measurement and l2 loss
                        measurement_losses['dcgan'][key] = 1.0#utils.get_measurement_loss(x_hat, A_outer, y)
                        l2_losses['dcgan'][key] = utils.get_l2_loss(x_hat, x)
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
                outputdir = 'res/PhaseRetrieval_CelabA/'
                if not os.path.exists(outputdir):
                    os.mkdir(outputdir)
                hparams.savepath=outputdir+'phaseretrieval_m_%d_r_%d.png'%(hparams.num_outer_measurements,ri)
                # np.savez(hparams.savepath[:-4]+'argsx.npz',argsx=np.sort(np.absolute(xw_est_batch),1))
                utils.image_matrix_mls(xs_dict, x_hats_dict, view_image, hparams)
                img_rec_ls=[]
                for mi in hparams.method_ls:
                    print(len(x_hats_dict),mi, x_hats_dict[mi].keys())
                    img_rec_ls = img_rec_ls+[np.stack([vi for vi in x_hats_dict[mi].values()] , axis=0)]
                    
                np.savez(hparams.savepath[:-4]+'.npz',img_gd=np.stack([vi for vi in xs_dict.values()] , axis=0),img_rec=np.stack(img_rec_ls, axis=0))                                      
                    
                        # if hparams.image_matrix > 0:
                        #     # hparams.method = 'PPower'
                        #     outputdir = 'res/SpikedCov_celabA/'
                        #     if not os.path.exists(outputdir):
                        #         os.mkdir(outputdir)
                        #     hparams.savepath=outputdir+'phaseretrieval_%s_m_%d_r_%d.png'%(hparams.method,hparams.num_outer_measurements,ri)
                        #     utils.image_matrix(xs_dict, x_hats_dict, view_image, hparams)
                        #     np.savez(hparams.savepath[:-4]+'.npz',img_gd=x_batch,img_rec=x_hat_batch)
                    
                        # # Warn the user that some things were not processsed
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
    PARSER.add_argument('--num-input-images', type=int, default=1, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=1, help='How many examples are processed together')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type: as of now supports only gaussian')
    PARSER.add_argument('--noise-std', type=float, default=0.1, help='std dev of noise')


    # Measurement type specific hparams

    PARSER.add_argument('--num-outer-measurement-ls', metavar='N', type=int, default=[300], nargs='+',
                    help='number of measurements') #type=int, default=500, help='number of gaussian measurements(outer)')
    PARSER.add_argument('--beta-ls', metavar='N', type=float, default=[300], nargs='+',
                    help='list of beta') #type=int, default=500, help='number of gaussian measurements(outer)')
    PARSER.add_argument('--method', type=str, default='PPower', help='PPower,Power,TPower')
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
    PARSER.add_argument('--max-update-iter', type=int, default=30, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=2, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')
    PARSER.add_argument('--outer-learning-rate', type=float, default=1.25, help='learning rate of outer loop GD')
    PARSER.add_argument('--max-outer-iter', type=int, default=100, help='maximum no. of iterations for outer loop GD')

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
