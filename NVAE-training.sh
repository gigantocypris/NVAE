#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J NAVE-training
#SBATCH --mail-user=gchen4@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -A m3562

export EXPR_ID=test_0000
export DATA_DIR=data
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=$(hostname)
        
#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=none  python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 4 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 \
        --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --use_nersc