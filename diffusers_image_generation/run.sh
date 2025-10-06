#!/bin/bash

accelerate launch /netscratch/muhammad/codes/diffusers_image_generation/train_unconditional.py \
  --resolution=64 --center_crop --random_flip \
  --output_dir="/netscratch/muhammad/codes/diffusers_image_generation/Results/extended" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --train_data_dir="/ds/images/microscopy_3D/extended_set_2D_z_only/images/"