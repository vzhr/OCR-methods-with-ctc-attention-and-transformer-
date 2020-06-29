#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3; python2.7 ./main.py --train_file=/data/notebooks/yuandaxing1/DM/OCR/recognition/dataset/0907_all.txt  --val_dir=  --image_height=32  --image_width=600  --image_channel=1  --out_channels=256  --num_hidden=512  --batch_size=100  --log_dir=./log/train  --mode=train --map_file=/data/notebooks/yuandaxing1/map1.txt --gpus=0,1,2,3 --restore=true

#export CUDA_VISIBLE_DEVICES=0; python ./main.py --train_file=/data/users/yiweizhu/school/school_train_img.txt  --val_dir=  --image_height=32  --image_width=800  --image_channel=1  --out_channels=512  --num_hidden=512  --batch_size=80  --log_dir=./log/train  --mode=train --map_file=/data/users/yiweizhu/ocr/new_chinese/map1.txt --gpus=0 --restore=True --checkpoint_dir=./checkpoint

export CUDA_VISIBLE_DEVICES=0; python ./main.py --train_file=  --infer_file=/data/users/yiweizhu/school/school_test_img.txt  --image_height=32  --image_width=800  --image_channel=1  --out_channels=512  --num_hidden=512  --batch_size=50  --log_dir=./log/train  --mode=infer --map_file=/data/users/yiweizhu/ocr/new_chinese/map1.txt --gpus=0 --restore=true --output_dir=/data/users/yiweizhu/school/predict_attention/  --checkpoint_dir=./checkpoint/ 





#export LD_LIBRARY_PATH=/data/notebooks/yuandaxing1/cuda-9.0/lib64:${LD_LIBRARY_PATH}
#export CUDA_VISIBLE_DEVICES=3
#python2.7 ./infer_debug.py --infer_file=/data/notebooks/zhuyiwei/img.txt   --image_height=32  --image_width=448  --image_channel=1  --out_channels=256  --num_hidden=512  --batch_size=20  --log_dir=./log/train --mode=infer --map_file=/data/notebooks/yuandaxing1/map1.txt --gpus=0,1,2,3 --output_dir=/data/notebooks/zhuyiwei/chinese_re_predict/

#python2.7 ./model_export.py --infer_file=/data/notebooks/yuandaxing1/OCR/CNN_LSTM_CTC_Tensorflow/testing/0620_test_list   --image_height=64  --image_width=640  --image_channel=1  --out_channels=256  --num_hidden=512  --batch_size=1  --log_dir=./log/train --mode=infer --map_file=/data/notebooks/yuandaxing1/map1.txt --gpus=3 --output_dir=./export/

#export LD_PRELOAD=/opt/conda/lib/libmkl_def.so:/opt/conda/lib/libmkl_avx.so:/opt/conda/lib/libmkl_core.so:/opt/conda/lib/libmkl_intel_lp64.so:/opt/conda/lib/libmkl_intel_thread.so:/opt/conda/lib/libiomp5.so; export CUDA_VISIBLE_DEVICES=3 ; python2.7 model_serving.py --infer_file=/data/notebooks/yuandaxing1/OCR/CNN_LSTM_CTC_Tensorflow/testing/0620_test_list   --image_height=32  --image_width=448  --image_channel=1  --out_channels=256  --num_hidden=512  --batch_size=5  --log_dir=./log/train --mode=infer --map_file=/data/notebooks/yuandaxing1/map1.txt --gpus=0
