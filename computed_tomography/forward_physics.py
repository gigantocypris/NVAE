"""
PyTorch implementation of the Radon transform
A real space stack of 2D images are each converted to a sinogram (line projections)
"""

import torch
import numpy as np
import kornia



def pad_phantom(phantom,
                ):
    
    '''
    phantom is batch_size x img_size_x x img_size_y x 1
    '''

    img_size_x = phantom.shape[1]
    img_size_y = phantom.shape[2]

    # img_size_z = phantom.shape[2]
    num_proj_pix = torch.sqrt(img_size_x**2 + img_size_y**2) + 2
    num_proj_pix = torch.ceil(num_proj_pix / 2.) * 2
    
    odd_x = (num_proj_pix-img_size_x)%2
    odd_y = (num_proj_pix-img_size_y)%2
        
    padx = (num_proj_pix-img_size_x)//2
    pady = (num_proj_pix-img_size_y)//2
    

    paddings = [[0,0],[padx, padx+odd_x], [pady, pady+odd_y],[0,0]]

    phantom = torch.nn.functional.pad(phantom, paddings, 'constant')
    return(phantom)


def project_torch(phantom, theta, pad = True, 
                  ):
    '''
    phantom is batch_size x img_size_x x img_size_y x 1

    rotation reference: https://kornia-tutorials.readthedocs.io/en/latest/_nbs/rotate_affine.html
    '''
    
    num_angles = len(theta)
    _, x, y, _ = phantom.shape

    if pad:
        phantom = pad_phantom(phantom)
    
    imgs = torch.transpose(phantom, perm=[3,1,2,0])
    imgs = imgs.repeat(num_angles) # XXX CHECK THIS IS THE SAME
    # imgs = tf.repeat(imgs, num_angles, axis=0)

    
    imgs_rot = kornia.geometry.rotate(imgs, -theta)
    sino = torch.sum(imgs_rot, 1)
    # imgs_rot = tfa.image.rotate(imgs,-theta)
    # sino = tf.math.reduce_sum(imgs_rot,1)
    

    # rotate back
    sino = torch.transpose(sino, perm=[2,0,1])
    
    # add back dummy dimension
    sino = sino.unsqueeze(-1)
    # sino = tf.expand_dims(sino, axis=-1)
        
    return(sino)