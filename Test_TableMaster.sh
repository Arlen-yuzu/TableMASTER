#! /bin/bash

python test.py --data_name icdar13 --data_abs_fold /shared/aia/alg/xyl/tsrdataset/unify/icdar13/ --out_dir ./test_result/hubtabnet_res/html --device cuda:1

#python test.py --data_name comfintab --data_abs_fold /shared/aia/alg/xyl/tsrdataset/unify/comfintab/ --out_dir ./test_result/comfintab_res/html --device cuda:1

#python test.py --data_name pubtabnet --data_abs_fold /shared/aia/alg/xyl/tsrdataset/unify/pubtabnet/ --out_dir ./test_result/pubtabnet_res/html --device cuda:1

#python test.py --data_name tablegraph24k --data_abs_fold /shared/aia/alg/xyl/tsrdataset/unify/tablegraph24k/ --out_dir ./test_result/tablegraph24k_res/html --device cuda:1

# python test.py --data_name icdar19 --data_abs_fold /shared/aia/alg/xyl/tsrdataset/unify/icdar19/ --out_dir ./test_result/icdar19_res/html --device cuda:1
