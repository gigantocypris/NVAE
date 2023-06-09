# Setup on Strelka:

Navigate to:
https://strelka.swarthmore.edu/

Interactive Apps --> JupyterLab

Choose the number of hours and start a session.

Once in the session, open a terminal.

In the terminal:
If needed, install miniconda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Log out and log back in.

Update conda:
```
conda update -n base -c defaults conda
```

Navigate to the directory where you want the code to reside and clone the repository.
```
git clone https://github.com/gigantocypris/NVAE.git
```

Create conda environment with Tomopy and install PyTorch 2.0: 
```
conda create --name tomopy --channel conda-forge tomopy python=3.9
conda activate tomopy
conda install pytorch==2.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Test PyTorch GPU installation:
```
python
import torch
```

```
print(torch.__version__)
```

```
print(torch.cuda.is_available())
```

```
print(torch.cuda.device_count())
```

Expected output:
```
2.0.0
True
1
```

```
quit()
```

Install the other conda dependencies:
```
conda install h5py
conda install xdesign -c conda-forge
```

Install the other pip dependencies:
```
pip install --upgrade pip
python -m pip install -r requirements.txt
```

To use the conda environment:
```
conda activate tomopy
```

To exit the conda environment:
```
conda deactivate
```

Run NVAE:

Navigate to the directory containing NVAE:
```
cd NVAE
mkdir data
mkdir checkpts
```

Training NVAE with MNIST, single GPU (for multiple GPUs, increase --num_process_per_node):
```
export EXPR_ID=test_0000
export DATA_DIR=data
export CHECKPOINT_DIR=checkpts
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax
```

Small number of epochs/smaller network
```
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 4 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 \
        --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax
```

# Training NVAE on NERSC

TODO: Setup directions (see project-management/notes_NVAE for now)

```
module load python 
conda activate NVAE_2
cd $HOME/NVAE

salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m2859_g --ntasks-per-gpu=1 --cpus-per-task=32

export EXPR_ID=test_0000
export DATA_DIR=data
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=$(hostname)

```

```
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 4 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 \
        --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --use_nersc
```


# Setup on MacBook Pro
This setup can only be used for generating data and/or running CT_PVAE, NVAE will not work on a MacBook Pro as an NVIDIA GPU is required.

Install Anaconda or miniconda if necessary.

Update conda:
```
conda update -n base -c defaults conda
```

Navigate to the directory where you want the code to reside and clone the repository.
```
git clone https://github.com/gigantocypris/NVAE.git
```

Create conda environment with Tomopy: 
```
conda create --name tomopy --channel conda-forge tomopy python=3.9
conda activate tomopy
```


Install the other conda dependencies:
```
conda install xdesign -c conda-forge
```

Install the other pip dependencies:
```
python -m pip install --upgrade pip
python -m pip install -r NVAE/requirements.txt
```

To exit the conda environment:
```
conda deactivate
```

# Dataset preparation

Activate your environment if not already activated:
```
conda activate tomopy
```

Create a working directory:
```
mkdir working_dir
```

Create an environment variable pointing to the NVAE directory:
```
export NVAE_PATH=path_to_NVAE
```

Navigate to the working directory
```
cd working_dir
```

Run the following to create a synthetic foam dataset of 50 examples, saved in the subfolder `dataset_foam` of the current directory:

```
python $NVAE_PATH/computed_tomography/create_foam_images.py -n 50
```

To visualize:
```
python
import numpy as np
foam_imgs = np.load('foam_training.npy')
import matplotlib.pyplot as plt
plt.imshow(foam_imgs[0,:,:]); plt.show()
```

To create sinograms from the foam images:
```
python $NVAE_PATH/computed_tomography/images_to_sinograms.py -n 50
```

TODO: create sparse sinograms
TODO: reconstruct from sparse sinograms

# TODO

version numbers for all packages in the install directions

