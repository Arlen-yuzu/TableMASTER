import os
from argparse import ArgumentParser

import torch
from mmcv.image import imread
import sys
import cv2
sys.path.append('/data/xuyilun/project/tsrbenchmark/models/TableMASTER')
from table_recognition.table_inference import Detect_Inference, Recognition_Inference, End2End, Structure_Recognition
from table_recognition.match import DemoMatcher

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401

import glob
import time
import pickle
import numpy as np
from tqdm import tqdm
import json
import jsonlines
from Evaluator import Evaluator
from cal_tasks import model_dataset_info
from html_utils import format_html
from res_visual import show_results


def four2eight(bbox):
    x1, y1, x3, y3 = bbox[0], bbox[1], bbox[2], bbox[3]
    return [x1, y1, x3, y1, x3, y3, x1, y3]

def read_files_in_folder(folder_path):
    # 列出目录下的所有文件和文件夹
    file_list = os.listdir(folder_path)
    file_path= []
    for file_name in file_list:
        #拼接目录名和文件名
        file_path.append(os.path.join(folder_path, file_name))
    return file_path

def get_img_name(img_path):
    return img_path.split('/')[-1].split('.')[0]

def clear_zeros(bboxs):
    c_bboxs = []
    for bbox in bboxs:
        if sum(bbox) == 0.: continue
        else:
            c_bboxs.append(four2eight_for_center(bbox))
    return np.array(c_bboxs)

def four2eight_for_center(bbox):
    #bbox = [x,y,w,h]
    x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
    x1, y1 = x - (w/2), y - (h/2)
    x3, y3 = x + (w/2), y + (h/2)
    return [x1, y1, x3,y1, x3,y3, x1,y3]

def get_label(ann):
    bbox = []
    for cell in ann['html']['cells']:
        bbox.append(four2eight(cell['bbox']))
    return np.array(bbox)

def htmlPostProcess(text):
    text = '<html><body><table>' + text + '</table></body></html>'
    return text

if __name__ == '__main__':
    parser = ArgumentParser()    
    parser.add_argument('--tablemaster_config', type=str,
                        default='./configs/textrecog/master/table_master_ResnetExtract_Ranger_0705.py',
                        help='tablemaster config file')
    parser.add_argument('--tablemaster_checkpoint', type=str,
                        default='./checkpoints/table_master.pth',
                        help='tablemaster checkpoint file')
    parser.add_argument('--out_dir',
                        type=str, default='/data/xuyilun/icdar13_res/html', help='Dir to save results')
    
    parser.add_argument('--data_name', type=str, default='tablegraph24k')                    
    parser.add_argument('--img_dir', type=str, default='/shared/aia/alg/xyl/tsrdataset/unify/tablegraph24k/image')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'

    cal_tasks = model_dataset_info[args.data_name]['cal_tasks']
    eval_fn = Evaluator(cal_tasks, match_type=model_dataset_info[args.data_name]['match'])

    # in process
    import sys
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    img_dir = args.img_dir
    
    # model
    tablemaster_inference = Structure_Recognition(args.tablemaster_config, args.tablemaster_checkpoint, device)
    
    test_demo_num = 0
    
    teds = []
    if args.data_name == 'pubtabnet':
        teds = {}
        with jsonlines.open('/data/xuyilun/project/LORE-TSR/data/pubtabnet_lore/PubTabNet_2.0.0_val.jsonl', 'r') as f:
            for tab in f:
                teds[tab['filename']] = format_html(tab)
    
    imgs = os.listdir(img_dir)
    if not os.path.exists("vis_hard_real"):
        os.makedirs("vis_hard_real")
    
    for img_name in tqdm(imgs):
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path): continue
        image = cv2.imread(img_path)
        
        # table structure predict
        tablemaster_result, _ = tablemaster_inference.predict_single_file(img_path)
        torch.cuda.empty_cache()
        
        pred_ted_list = tablemaster_result['text'].split(',')
        pred_ted_list_new = []
        for i in range(len(pred_ted_list)):
            if pred_ted_list[i] == '<td></td>':
                pred_ted_list_new.append('<td>')
                pred_ted_list_new.append('</td>')
            else:
                pred_ted_list_new.append(pred_ted_list[i])
                
        pred_bboxs = clear_zeros(tablemaster_result['bbox'])
        preds = {'pred_bbox': pred_bboxs, 'pred_lloc': np.empty((0, 4))}
        
        if len(pred_bboxs):
            show_results(img_name, image, preds['pred_bbox'], preds['pred_lloc'], "vis_hard_real")
        
        if args.data_name == 'pubtabnet':
            preds['pred_html'] = format_html(pred_ted_list_new)
        elif args.data_name == 'sym':
            preds['pred_html'] = format_html(pred_ted_list_new)
        