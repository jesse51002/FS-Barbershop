#!/usr/bin/env bash

#python main.py --im_path1 short-wavy.png --im_path2 braids.png --im_path3 braids.png \
#    --sign realistic  --smooth 5 --learning_rate 0.01 \
#    --align_steps1 50 --align_steps2 50 --blend_steps 200 \
#    --W_steps=1000 --FS_steps=200  \
#    --hair_perc_lambda 0 --body_alternate_number 5 --opt_name adam

#python main.py --im_path1 IMG_3248.png --im_path2 shoulder_length_wavy_wispy_bangs.png --im_path3 shoulder_length_wavy_wispy_bangs.png \
#    --sign realistic  --smooth 5 --learning_rate 0.05 \
#    --align_steps1 80 --align_steps2 120 --blend_steps 100 \
#    --W_steps=300 --FS_steps=150  \
#    --style_lambda 200000 --body_alternate_number 5 --align_color_lambda 0.0001

    
python main.py --im_path1 IMG_3248.png --im_path2 shoulder_length_wavy_wispy_bangs.png --im_path3 shoulder_length_wavy_wispy_bangs.png \
    --model StyleGanXL --ckpt 'pretrained_models/ffhq1024xl.pkl'\
    --sign realistic  --smooth 5 --learning_rate 0.05 \
    --align_steps1 1 --align_steps2 1 --blend_steps 1 \
    --W_steps=200 --FS_steps=1  \
    --style_lambda 200000 --body_alternate_number 5 --align_color_lambda 0.0001