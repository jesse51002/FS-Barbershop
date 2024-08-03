#!/usr/bin/env bash

#python main.py --im_path1 short-wavy.png --im_path2 braids.png --im_path3 braids.png \
#    --sign realistic  --smooth 5 --learning_rate 0.01 \
#    --align_steps1 50 --align_steps2 50 --blend_steps 200 \
#    --W_steps=1000 --FS_steps=200  \
#    --hair_perc_lambda 0 --body_alternate_number 5 --opt_name adam

python main.py --im_path1 IMG_4302.png --im_path2 werewolf.png --im_path3 werewolf.png \
    --sign realistic  --smooth 5 --learning_rate 0.05 \
    --align_steps1 80 --align_steps2 120 --blend_steps 100 \
    --W_steps=300 --FS_steps=150  \
    --style_lambda 200000 --body_alternate_number 5 --align_color_lambda 0.0001 \
    --clip_quality --clip_quality_iterations 20 \
    --hair_class 2 --hair_classifier_iterations 200 --hair_type_lambda 0.4


# Time with no clip: 13 seconds
# Time with 20 iterations clip:  15 seconds
