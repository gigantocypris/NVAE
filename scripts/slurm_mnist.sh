#!/bin/bash

#SBATCH -N 2            # Number of nodes
#SBATCH -J NVAE      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m2859_g       # allocation
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:36:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=2
#SBATCH -o %j.out
#SBATCH -e %j.err

export EXPR_ID=test_0000_slurm
export DATA_DIR=data
export CHECKPOINT_DIR=checkpts

echo "jobstart $(date)";pwd


srun -n 8 -G 8 -c 2 python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 8 --use_se --res_dist --fast_adamax 
        
echo "jobend $(date)";pwd