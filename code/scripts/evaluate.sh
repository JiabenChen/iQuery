#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id " #PUT the model id here
OPTS+="--list_train iQuery/data/MUSIC/trainmusic.csv "
OPTS+="--list_val iQuery/data/MUSIC/testsepmusic.csv "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "
OPTS+="--split test "

#maskformer settings
OPTS+="--in_channels 256 " #"channels of the input features"
OPTS+="--MASK_FORMER_HIDDEN_DIM 256 " #"hidden_dim"
OPTS+="--MASK_FORMER_NUM_OBJECT_QUERIES 12 " #"num_queries"
OPTS+="--MASK_FORMER_NHEADS 8 " #"nheads"
OPTS+="--MASK_FORMER_DROPOUT 0 " #dropout
OPTS+="--MASK_FORMER_DIM_FEEDFORWARD 1024 " #"dim_feedforward"
OPTS+="--MASK_FORMER_ENC_LAYERS 1 "
OPTS+="--MASK_FORMER_DEC_LAYERS 4 " #"dec_layers"
OPTS+="--SEM_SEG_HEAD_MASK_DIM 32 " #"mask_dim"

# loss
OPTS+="--binary_mask 0 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 1 "

# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 1 "

python -u main_music.py $OPTS
