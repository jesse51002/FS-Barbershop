#!/usr/bin/env bash

python main.py --im_path1 straightlongbobhairstyle8.png --im_path2 straightlongbobhairstyle34.png --im_path3 straightlongbobhairstyle34.png \
    --sign realistic  --smooth 5 --learning_rate 0.1 \
    --align_steps1 10 --align_steps2 50 --blend_steps 200 \
    --W_steps=100 --FS_steps=150  \
    --hair_perc_lambda 0 --style_lambda 200000 --body_alternate_number 5