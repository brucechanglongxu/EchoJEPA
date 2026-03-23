#!/bin/bash
# SonoState training on 7 GPUs (skipping GPU 0 which is in use)
# Uses nohup so training survives SSH disconnects

cd /scratch/bxu/project/EchoJEPA

export PYTHONPATH=/scratch/bxu/project/EchoJEPA

nohup python3 -m app.main \
    --fname configs/train/vitl16/sonostate-mimic-224px-16f.yaml \
    --devices cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    > /scratch/bxu/project/EchoJEPA/checkpoints/sonostate_vitl_224px_16f/train_stdout.log 2>&1 &

echo "SonoState training launched with PID: $!"
echo "Monitor with: tail -f /scratch/bxu/project/EchoJEPA/checkpoints/sonostate_vitl_224px_16f/train_stdout.log"
