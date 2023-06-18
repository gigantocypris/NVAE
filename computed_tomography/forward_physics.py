"""
PyTorch implementation of the Radon transform
A real space stack of 2D images are each converted to a sinogram (line projections)
"""
import numpy as np
import torch
import kornia

def pad_phantom(phantom,
                ):
    '''
    phantom is batch_size x img_size_x x img_size_y x 1
    '''
    img_size_x = phantom.shape[1]
    img_size_y = phantom.shape[2]

    num_proj_pix = np.sqrt(img_size_x**2 + img_size_y**2) + 2
    num_proj_pix = np.ceil(num_proj_pix / 2.) * 2
    
    odd_x = int((num_proj_pix-img_size_x)%2)
    odd_y = int((num_proj_pix-img_size_y)%2)
        
    padx = int((num_proj_pix-img_size_x)//2)
    pady = int((num_proj_pix-img_size_y)//2)
    
    paddings = (0,0,pady,pady+odd_y,padx,padx+odd_x,0,0)
    phantom = torch.nn.functional.pad(phantom, paddings, 'constant',value=0)
    return(phantom)


def project_torch(phantom, theta_degrees, pad=True, 
                  ):
    '''
    phantom is batch_size x img_size_x x img_size_y x 1
    rotation tutorial: https://kornia-tutorials.readthedocs.io/en/latest/_nbs/rotate_affine.html
    rotation reference: https://kornia.readthedocs.io/en/latest/geometry.transform.html
    '''
    breakpoint()
    ### STOPPED HERE
    num_angles = len(theta_degrees)
    if pad:
        phantom = pad_phantom(phantom)
    
    phantom = phantom.repeat(1,1,1,num_angles)
    phantom = torch.transpose(phantom, 2,3)
    phantom = torch.transpose(phantom, 1,2)
    phantom = torch.transpose(phantom, 0,1)
    
    imgs_rot = kornia.geometry.rotate(phantom, -theta_degrees)
    sino = torch.sum(imgs_rot, 2)
    
    # transpose back
    sino = torch.transpose(sino, 0, 1)

    return(sino)