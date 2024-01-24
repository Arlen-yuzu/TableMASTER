#! /bin/bash
python test.py \
    --data_name pubtabnet \
    --img_dir /shared/aia/alg/xyl/tsrdataset/unify/pubtabnet/image \
    --ann_path /shared/aia/alg/xyl/tsrdataset/model/tablemaster/pubtabnet/val.jsonl \
    --tablemaster_checkpoint /data/xuyilun/project/TableMASTER/work_dir/0115_sym/latest.pth \
    --out_dir ./test_result/pubtabnet_res/html \
    --device cuda:7