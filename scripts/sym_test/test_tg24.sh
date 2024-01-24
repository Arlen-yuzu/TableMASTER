#! /bin/bash
python test.py \
    --data_name tablegraph24k \
    --img_dir /shared/aia/alg/xyl/tsrdataset/unify/tablegraph24k/image \
    --ann_path /shared/aia/alg/xyl/tsrdataset/model/tablemaster/tablegraph24k/test.jsonl \
    --tablemaster_checkpoint /data/xuyilun/project/TableMASTER/work_dir/0115_sym/latest.pth \
    --out_dir ./test_result/pubtabnet_res/html \
    --device cuda:7