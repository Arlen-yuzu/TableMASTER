CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh \
    ./configs/textrecog/master/table_master_lmdb_ResnetExtract_Ranger_0930.py \
    ./work_dir/1114_TableMASTER_structure/ \
    8
