#!/bin/bash

export CUDA_VISIBLE_DEVICES=2;python ./main.py
#export CUDA_VISIBLE_DEVICES=0; python ./main.py --train_file=/data/users/yiweizhu/ocr/new_chinese/train_img.txt  --val_dir=  --image_height=32  --image_width=800  --image_channel=1  --out_channels=256  --hidden_size=512  --batch_size=80  --log_dir=./log/train  --mode=train --map_file=/data/users/yiweizhu/ocr/new_chinese/map.txt --gpus=0 --restore=True --checkpoint_dir=./checkpoint

#export CUDA_VISIBLE_DEVICES=0; python ./main.py --train_file=  --infer_file=/data/users/yiweizhu/school/school_test_img.txt  --image_height=32  --image_width=800  --image_channel=1  --out_channels=512  --num_hidden=512  --batch_size=1  --log_dir=./log/train  --mode=infer --map_file=/data/users/yiweizhu/ocr/new_chinese/map1.txt --gpus=0 --restore=true --output_dir=/data/users/yiweizhu/school/predict/  --checkpoint_dir=./checkpoint2/ 



