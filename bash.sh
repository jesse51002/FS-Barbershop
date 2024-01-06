#!/usr/bin/env bash

python main.py --im_path1 j.png --im_path2 miniafrohairstyle11.png --im_path3 miniafrohairstyle11.png \
    --sign realistic  --smooth 5 --learning_rate 0.1 \
    --align_steps1 20 --align_steps2 20 --blend_steps 20 \
    --W_steps=200 --FS_steps=100 