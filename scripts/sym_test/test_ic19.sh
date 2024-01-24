#! /bin/bash
python test.py \
    --data_name icdar19 \
    --img_dir /shared/aia/alg/xyl/tsrdataset/unify/icdar19/image \
    --ann_path /shared/aia/alg/xyl/tsrdataset/model/tablemaster/icdar19/test.jsonl \
    --tablemaster_checkpoint /data/xuyilun/project/TableMASTER/work_dir/0115_sym/latest.pth \
    --out_dir ./test_result/pubtabnet_res/html \
    --device cuda:7