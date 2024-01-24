#! /bin/bash
python test.py \
    --data_name sym \
    --img_dir /shared/aia/alg/xyl/tsrdataset/unify/sym/image \
    --ann_path /shared/aia/alg/xyl/tsrdataset/model/tablemaster/sym/test.jsonl \
    --tablemaster_checkpoint /data/xuyilun/project/TableMASTER/work_dir/0115_sym/latest.pth \
    --out_dir ./test_result/vis_sym \
    --device cuda:6