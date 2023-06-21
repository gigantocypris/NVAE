# Setup on remote server:

Navigate to the directory (Documents) where you want the code to reside and clone the repository.
```
git clone https://github.com/gigantocypris/NVAE.git
```

Create conda environment with Tomopy: 
```
conda create --name tomopy --channel conda-forge tomopy=1.14.1 python=3.9
conda activate tomopy
```
This environment will be used for all Tomopy preprocessing.

Activate the PINN environment:
```
conda activate PINN
```

Install the other pip dependencies in the PINN environment:
```
pip install --upgrade pip
python -m pip install -r requirements.txt
```


Run NVAE in the PINN environment:

Navigate to the directory containing NVAE:
```
cd NVAE
mkdir data
mkdir checkpts
```

Training NVAE with MNIST, 4 GPUs (number of GPUs is in --num_process_per_node):
```
export EXPR_ID=test_0000
export DATA_DIR=data
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=localhost
```

Full MNIST example:
```
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 --epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax
```

Small number of epochs/smaller network for MNIST:
```
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 --epochs 2 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax
```

Launch Tensorboard to view results:
tensorboard --logdir $CHECKPOINT_DIR/eval-$EXPR_ID/

Single GPU for debugging: (MNIST)
```
export EXPR_ID=test_0000
export DATA_DIR=data
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=localhost

python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 --epochs 2 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax
```

Foam dataset, single GPU:
```
export EXPR_ID=test_0000
export DATA_DIR=data
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=localhost

python train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam --batch_size 8 --epochs 2 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax
```


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
conda create --name tomopy3 --channel conda-forge tomopy python=3.9
conda activate tomopy3
conda install xdesign -c conda-forge
```

Separate conda environment for Pytorch:
```
conda create --name pytorch python=3.9
conda activate pytorch
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
```

Install the other pip dependencies:
```
pip install --upgrade pip
python -m pip install -r requirements.txt
```

To use the conda environment:
```
conda activate tomopy3
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
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 --epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax
```

Small number of epochs/smaller network
```
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 --epochs 4 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax
```


# Setup on NERSC

TODO: Setup directions should be updated to include the packages in the directions for Strelka

```
cd $HOME
module load python
conda create -n NVAE_2 python=3.7 -y
conda activate NVAE_2
git clone https://github.com/gigantocypris/NVAE.git
cd NVAE
```

```
mkdir data
mkdir checkpts
```

```
conda install pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
python -m pip install scipy
```

# Training NVAE on NERSC

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

Single GPU training:
```
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 4 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 \
        --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --use_nersc
```

Multi GPU training:
```
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 4 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 \
        --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --use_nersc
```

TODO: Try Tensorboard on NERSC

# Setup on MacBook Pro

This setup can only be used for generating data, NVAE will not work on a MacBook Pro as an NVIDIA GPU is required.

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

Create a working directory (e.g. Dropbox/output_CT_NVAE):
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
python $NVAE_PATH/computed_tomography/create_foam_images.py -t 10 -v 10
```

To visualize:
```
python
import numpy as np
foam_imgs = np.load('foam_training.npy')
import matplotlib.pyplot as plt
plt.imshow(foam_imgs[0,:,:]); plt.show()
```

To create sinograms from the foam images, create sparse sinograms, and reconstruct from the sparse sinograms:
```
python $NVAE_PATH/computed_tomography/images_to_dataset.py -n 10 -d train
```
(If needed, `export KMP_DUPLICATE_LIB_OK=TRUE`)

Test forward_physics:
```
python $NVAE_PATH/computed_tomography/test_forward_physics.py 
```

# TODO

version numbers for all packages in the install directions


/global/homes/X/XX/.conda/envs/NVAE_2/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py:2388: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.