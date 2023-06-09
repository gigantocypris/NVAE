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

Small number of epochs
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 4 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax


# Create the dataset using CT_PVAE

# Add an option to NVAE to train the CT_NVAE

# Get final metrics

# Setup on MacBook Pro

Install Anaconda or miniconda if necessary.

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
conda install pytorch::pytorch torchvision torchaudio -c pytorch
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
pip install --upgrade tensorflow-probability
```

To use the conda environment:
```
conda activate tomopy
```

To exit the conda environment:
```
conda deactivate
```