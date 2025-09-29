#srun -K -p A100-80GB,H100,H200,H200-SDS,H100-RP \
#batch,V100-32GB-SDS,V100-32GB,V100-32GB-SDS,RTXA6000 
srun -K -p A100-80GB,H100,H200,H200-SDS,H100-RP  \
  --job-name="Gen2SegwithBG" \
  --gpus=1 \
  --ntasks=1 \
  --time=00-2:00 \
  --cpus-per-task=8 \
  --container-image=/netscratch/muhammad/gen2seg2.sqsh  \
  --container-workdir="`pwd`" \
  --mem=64G \
  --pty \
  --immediate=3400 \
  --container-mounts=/netscratch:/netscratch,"`pwd`":"`pwd`" \
    bash
#    /bin/bash -c "bash /netscratch/muhammad/codes/gen2seg/training/scripts/train_sd_fluoDataset.sh"
