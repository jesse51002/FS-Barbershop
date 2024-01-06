#!/usr/bin/env bash

python main.py --im_path1 curlyshortbobhairstyle34.png --im_path2 miniafrohairstyle11.png --im_path3 miniafrohairstyle11.png \
    --sign realistic  --smooth 5 --learning_rate 0.1 \
    --align_steps1 10 --align_steps2 20 --blend_steps 150 \
    --W_steps=200 --FS_steps=100  \
    --hair_perc_lambda 0