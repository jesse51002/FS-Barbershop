#!/usr/bin/env bash

python main.py --im_path1 IMG_4305.png --im_path2 IMG_4305.png --im_path3 IMG_4305.png \
    --sign realistic  --smooth 5 --learning_rate 0.01 \
    --align_steps1 50 --align_steps2 50 --blend_steps 200 \
    --W_steps=1000 --FS_steps=200  \
    --hair_perc_lambda 0 --body_alternate_number 5 --opt_name adam


#python main.py --im_path1 base.png --im_path2 afro_tex_curls.png --im_path3 afro_tex_curls.png \
#    --sign realistic  --smooth 5 --learning_rate 0.05 \
#    --align_steps1 10 --align_steps2 50 --blend_steps 100 \
#    --W_steps=100 --FS_steps=150  \
#    --hair_perc_lambda 0 --style_lambda 200000 --body_alternate_number 5