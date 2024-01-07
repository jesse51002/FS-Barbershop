#!/usr/bin/env bash

python main.py --im_path1 curlyshortbobhairstyle34.png --im_path2 straightlongbobhairstyle8.png --im_path3 straightlongbobhairstyle8.png \
    --sign realistic  --smooth 5 --learning_rate 0.1 \
    --align_steps1 80 --align_steps2 1 --blend_steps 150 \
    --W_steps=100 --FS_steps=100  \
    --hair_perc_lambda 0 --style_lambda 200000