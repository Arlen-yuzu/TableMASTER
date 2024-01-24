#!/bin/bash
export CUDA_VISIBLE_DEVICES="4,5" 
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/xuyilun/opencv-3.4.10/lib

port=`shuf -i 29500-29600 -n1`
res=`lsof -i:${port}`
while [[ -n ${res} ]]; do
    port=$((port + 1))
    res=`lsof -i:${port}`
done

PORT=${PORT:-${port}}

echo $PORT

BASE_PATH=/data/xuyilun/project/TableMASTER
DATA_PATH=/shared/aia/alg/xyl/tsrdataset

CONFIG=$BASE_PATH/configs/tablemaster/sym.py
WORK_DIR=$BASE_PATH/shared/aia/alg/xyl/tsrckpt/tablemaster/0123_sym/
GPUS=2

alphabet_file=$DATA_PATH/model/tablemaster/sym/structure_alphabet.txt
train_img_prefix=$DATA_PATH/unify/sym/image
train_anno_file1=$DATA_PATH/model/tablemaster/sym/sym_lmdb/StructureLabel_train
valid_img_prefix=$train_img_prefix
valid_anno_file1=$DATA_PATH/model/tablemaster/sym/sym_lmdb/StructureLabel_val
test_img_prefix=$train_img_prefix
test_anno_file1=$DATA_PATH/model/tablemaster/sym/sym_lmdb/StructureLabel_test


if [ ${GPUS} == 1 ]; then
    python $BASE_PATH/tools/train.py  $CONFIG --work_dir=${WORK_DIR} \
        --batch_size 14 \
        --load-from $BASE_PATH/checkpoints/table_master.pth \
        --cfg-options \
        alphabet_file=$alphabet_file \
        train_img_prefix=$train_img_prefix \
        train_anno_file1=$train_anno_file1 \
        valid_img_prefix=$valid_img_prefix \
        valid_anno_file1=$valid_anno_file1 \
        test_img_prefix=$test_img_prefix \
        test_anno_file1=$test_anno_file1
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $BASE_PATH/tools/train.py $CONFIG --work_dir=${WORK_DIR} \
        --batch_size 8 \
        --load-from $BASE_PATH/checkpoints/table_master.pth \
        --launcher pytorch \
        --cfg-options \
        alphabet_file=$alphabet_file \
        train_img_prefix=$train_img_prefix \
        train_anno_file1=$train_anno_file1 \
        valid_img_prefix=$valid_img_prefix \
        valid_anno_file1=$valid_anno_file1 \
        test_img_prefix=$test_img_prefix \
        test_anno_file1=$test_anno_file1
fi