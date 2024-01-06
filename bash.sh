#!/usr/bin/env bash

python main.py --im_path1 damat.png --im_path2 mediumlengthtwostrandtwisthairstyle34.png --im_path3 mediumlengthtwostrandtwisthairstyle34.png \
    --sign realistic --smooth 5 --learning_rate 0.05 \
    --align_steps1 40 --align_steps2 40 --blend_steps 20 \
    --W_steps=100 --FS_steps=50 


https://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@github.com/jesse51002/FS-Barbershop.git