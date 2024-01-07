#!/usr/bin/env bash

python main.py --im_path1 mediumlengthtwostrandtwisthairstyle34.png --im_path2 miniafrohairstyle11.png --im_path3 miniafrohairstyle11.png \
    --sign realistic  --smooth 5 --learning_rate 0.1 \
    --align_steps1 20 --align_steps2 40 --blend_steps 150 \
    --W_steps=100 --FS_steps=100  \
    --hair_perc_lambda 100 --style_lambda 40000