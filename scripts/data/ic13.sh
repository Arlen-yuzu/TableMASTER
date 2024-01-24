#! /bin/bash
BASE_PATH=$(dirname $(dirname $(dirname $(realpath $0))))/table_recognition

splits=(train test)

for split in ${splits[@]}; do
    python $BASE_PATH/data_preprocess.py \
        --dataset_name icdar13 \
        --split $split \
        --nproc 16 \
        --img_base /shared/aia/alg/xyl/tsrdataset/unify \
        --ann_base /shared/aia/alg/xyl/tsrdataset/model/tablemaster
    
    python $BASE_PATH/lmdb_maker.py \
        --split $split \
        --lmdb-root /shared/aia/alg/xyl/tsrdataset/model/tablemaster/icdar13/icdar13_lmdb \
        --img-root /shared/aia/alg/xyl/tsrdataset/unify/icdar13/image \
        --txt_folder /shared/aia/alg/xyl/tsrdataset/model/tablemaster/icdar13/StructureLabelAddEmptyBbox_$split
done
