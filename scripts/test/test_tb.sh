#! /bin/bash
python test.py \
    --data_name taobao \
    --img_dir /shared/aia/alg/xyl/tsrdataset/unify/taobao/image \
    --ann_path /shared/aia/alg/xyl/tsrdataset/model/tablemaster/taobao/test.jsonl \
    --tablemaster_checkpoint /data/xuyilun/project/TableMASTER/checkpoints/table_master.pth \
    --out_dir ./test_result/pubtabnet_res/html \
    --device cuda:7