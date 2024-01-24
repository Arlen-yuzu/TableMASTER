#! /bin/bash
python test.py \
    --data_name comfintab \
    --img_dir /shared/aia/alg/xyl/tsrdataset/unify/comfintab/image \
    --ann_path /shared/aia/alg/xyl/tsrdataset/model/tablemaster/comfintab/test.jsonl \
    --tablemaster_checkpoint /data/xuyilun/project/TableMASTER/work_dir/0115_sym/latest.pth \
    --out_dir ./test_result/comfintab_res/html \
    --device cuda:7