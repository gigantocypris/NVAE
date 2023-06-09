import tomopy
import numpy as np
import os


def create_folder(save_path=None,**kwargs):
    try: 
        os.makedirs(save_path)
    except OSError:
        if not os.path.isdir(save_path):
            raise

def create_sinogram(img, theta, pad=True):
    # multiprocessing.freeze_support()
    phantom = np.expand_dims(img,axis=0)
    proj = tomopy.project(phantom, theta, center=None, emission=True, pad=pad, sinogram_order=False)
    proj = np.squeeze(proj,axis=1)
    return proj

def get_images(img_type = 'foam'):
    x_train = np.load(img_type + '_training.npy')
    return(x_train)