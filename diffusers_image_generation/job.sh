srun -K -p A100-80GB,H100,H200,H200-SDS,H100-RP \
  --job-name="imageGeneration" \
  --gpus=1 \
  --ntasks=1 \
  --time=00-10:00 \
  --cpus-per-task=8 \
  --container-image=/netscratch/muhammad/image_gen.sqsh  \
  --container-workdir="`pwd`" \
  --mem=64G \
  --container-mounts=/netscratch:/netscratch,"`pwd`":"`pwd`" \
    /bin/bash -c "bash /netscratch/muhammad/codes/diffusers_image_generation/run.sh"
