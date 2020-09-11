#!/bin/bash
savepath='/mnt/WDMyBook/RTML/beam-selection/out/img/'
echo $exp
echo "Start JOB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
python -u main.py \
    --data_folder /mnt/WDMyBook/RTML/beam-selection/baseline_data/ \
    --input 'lidar' \
    --shuffle True \
    --id_gpu 3 \
    --strategy one_hot \
    --epochs 15 \
    --lr 0.0001 \
    --Aug False \
    --augmented_folder /mnt/WDMyBook/RTML/beam-selection/aug_data_bu/ \
    >$savepath/lidar.out