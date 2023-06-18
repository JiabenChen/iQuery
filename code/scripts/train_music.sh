#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train ../data/MUSIC/trainmusic.csv "
OPTS+="--list_val ../data/MUSIC/testsepmusic.csv "

OPTS+="--split test "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "

# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 0 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 1 "
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

#maskformer decoder parameters
OPTS+="--in_channels 256 " #"channels of the input features"
OPTS+="--MASK_FORMER_HIDDEN_DIM 256 " #"hidden_dim"
OPTS+="--MASK_FORMER_NUM_OBJECT_QUERIES 12 " #"num_queries"
OPTS+="--MASK_FORMER_NHEADS 8 " #"nheads"
OPTS+="--MASK_FORMER_DROPOUT 0 " #dropout
OPTS+="--MASK_FORMER_DIM_FEEDFORWARD 1024 " #"dim_feedforward"
OPTS+="--MASK_FORMER_ENC_LAYERS 1 "
OPTS+="--MASK_FORMER_DEC_LAYERS 4 " #"dec_layers"
OPTS+="--SEM_SEG_HEAD_MASK_DIM 32 " #"mask_dim"
OPTS+="--lr_maskformer 0.0001 "
OPTS+="--weight_decay_maskformer 0.0001 "

OPTS+="--lr_drop_maskformer 80 "


# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 1 "


# learning params
OPTS+="--num_gpus 8 "
OPTS+="--workers 8 "
OPTS+="--batch_size_per_gpu 8 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-4 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 30 50 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

python -u main_music.py $OPTS
