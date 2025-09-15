#srun -K -p A100-80GB,H100,H200,H200-SDS,H100-RP \
#V100-32GB-SDS,V100-32GB,V100-32GB-SDS,RTXA6000
srun -K -p A100-80GB,H100,H200,H200-SDS,H100-RP \
  --job-name="Gen2SegwitheN3DHDistanceBasedWeights" \
  --gpus=1 \
  --ntasks=1 \
  --time=00-10:00 \
  --cpus-per-task=16 \
  --container-image=/netscratch/muhammad/gen2seg2.sqsh  \
  --container-workdir="`pwd`" \
  --mem=128G \
  --container-mounts=/netscratch:/netscratch,$(pwd):$(pwd),/ds \
    /bin/bash -c "bash /netscratch/muhammad/codes/gen2seg/training/scripts/train_sd_fluo_distance.sh"
     #/bin/bash -c "python /netscratch/muhammad/codes/gen2seg/debugging.py"
