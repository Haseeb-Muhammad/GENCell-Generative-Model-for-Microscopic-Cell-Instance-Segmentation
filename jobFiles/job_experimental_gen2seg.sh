#srun -K -p A100-80GB,H100,H200,H200-SDS,H100-RP \
srun -K -p A100-80GB,H100,H200,H200-SDS,H100-RP \
  --job-name="Gen2SegwithBG" \
  --gpus=1 \
  --ntasks=1 \
  --time=00-10:00 \
  --cpus-per-task=8 \
  --container-image=/netscratch/muhammad/gen2seg2.sqsh  \
  --container-workdir="`pwd`" \
  --mem=64G \
  --container-mounts=/netscratch:/netscratch,"`pwd`":"`pwd`" \
   /bin/bash -c "bash /netscratch/muhammad/codes/gen2seg_experimental/training/scripts/train_sd_fluoDataset.sh"
#  /bin/bash -c "python /netscratch/muhammad/codes/DataProcessing/get_neighbors.py"
