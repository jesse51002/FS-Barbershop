#!/usr/bin/env bash

python main.py --im_path1 heart.png --im_path2 curlyshortbobhairstyle34.png --im_path3 curlyshortbobhairstyle34.png \
    --sign realistic  --smooth 5 --learning_rate 0.05 \
    --align_steps1 10 --align_steps2 50 --blend_steps 100 \
    --W_steps=100 --FS_steps=150  \
    --hair_perc_lambda 0 --style_lambda 200000 --body_alternate_number 5