#!/bin/bash

accelerate launch /netscratch/muhammad/codes/gen2seg/training/train_sd_extended.py \
  --pretrained_model_name_or_path "reachomk/gen2seg-sd" \
  --checkpointing_steps 10000 \
  --train_batch_size 2  \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 6e-05 \
  --lr_total_iter_length 40000 \
  --lr_exp_warmup_steps 100 \
  --mixed_precision "no" \
  --num_train_epochs 25 \
  --output_dir "/netscratch/muhammad/codes/gen2seg/training/model-finetuned/stable_diffusion_extendedDatasetAllSlicesExtendedTraining_finetuning" \
  --enable_xformers_memory_efficient_attention \
  "$@"