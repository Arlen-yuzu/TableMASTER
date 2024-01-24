#! /bin/bash
BASE_PATH=$(dirname $(dirname $(dirname $(realpath $0))))/table_recognition

splits=(train val test)

for split in ${splits[@]}; do
    python $BASE_PATH/data_preprocess.py \
        --dataset_name sym \
        --split $split \
        --nproc 1 \
        --img_base /shared/aia/alg/xyl/tsrdataset/unify \
        --ann_base /shared/aia/alg/xyl/tsrdataset/model/tablemaster
    
    python $BASE_PATH/lmdb_maker.py \
        --split $split \
        --lmdb-root /shared/aia/alg/xyl/tsrdataset/model/tablemaster/sym/sym_lmdb \
        --img-root /shared/aia/alg/xyl/tsrdataset/unify/sym/image \
        --txt_folder /shared/aia/alg/xyl/tsrdataset/model/tablemaster/sym/StructureLabelAddEmptyBbox_$split
done
