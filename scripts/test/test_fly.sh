#! /bin/bash
python infer.py \
    --data_name icdar13 \
    --img_dir /data/xuyilun/project/hardreal \
    --tablemaster_checkpoint /data/xuyilun/project/TableMASTER/work_dir/0115_sym/latest.pth \
    --out_dir ./test_result/fly_res/html \
    --device cuda:4