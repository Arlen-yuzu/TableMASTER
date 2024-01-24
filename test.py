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
    # parser.add_argument('--pse_config', type=str,
    #                     default='./configs/textdet/psenet/psenet_r50_fpnf_600e_pubtabnet.py',
    #                     help='pse config file')
    # parser.add_argument('--pse_checkpoint', type=str,
    #                     default='./checkpoints/pse_epoch_600_wzw.pth',
    #                     help='pse checkpoint file')
    
    # parser.add_argument('--master_config', type=str,
    #                     default='./configs/textrecog/master/master_ResnetExtra_tableRec_dataset_dynamic_mmfp16.py',
    #                     help='master config file')
    # parser.add_argument('--master_checkpoint', type=str,
    #                     default='./checkpoints/master_epoch_6.pth',
    #                     help='master checkpoint file')
    
    parser.add_argument('--tablemaster_config', type=str,
                        default='./configs/textrecog/master/table_master_ResnetExtract_Ranger_0705.py',
                        help='tablemaster config file')
    parser.add_argument('--tablemaster_checkpoint', type=str,
                        default='./checkpoints/table_master.pth',
                        help='tablemaster checkpoint file')
    parser.add_argument('--out_dir',
                        # type=str, default='./test_result/tablegraph24k_res/html', help='Dir to save results')
                        #type=str, default='/data/xuyilun/comfintab_res/html', help='Dir to save results')
                        #type=str, default='/data/xuyilun/hubtabnet_res/html', help='Dir to save results')
                        #type=str, default='/data/xuyilun/hubtabnet_res/html', help='Dir to save results')
                        type=str, default='/data/xuyilun/icdar13_res/html', help='Dir to save results')
                        #type=str, default='/data/xuyilun/TM_tablegraph24k_res/html', help='Dir to save results')
    
    parser.add_argument('--data_name', type=str, default='tablegraph24k')                    
    parser.add_argument('--img_dir', type=str, default='/shared/aia/alg/xyl/tsrdataset/unify/tablegraph24k/image')
    parser.add_argument('--ann_path', type=str, default='/shared/aia/alg/xyl/tsrdataset/model/tablemaster/icdar13/test.jsonl')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    device = args.device if torch.cuda.is_available() else 'cpu'

    cal_tasks = model_dataset_info[args.data_name]['cal_tasks']
    eval_fn = Evaluator(cal_tasks, match_type=model_dataset_info[args.data_name]['match'])

    # in process
    import sys
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    img_dir = args.img_dir
    # text line detection and recognition end2end predict    
    #abs_fold = '/shared/aia/alg/xyl/tsrdataset/unify/icdar13/'
    #abs_fold = '/shared/aia/alg/xyl/tsrdataset/unify/comfintab/'
    #abs_fold = '/shared/aia/alg/xyl/tsrdataset/unify/tablegraph24k/'
    #abs_fold = '/fm1/wangziwei/wzw/TGRNet-main/datasets/tablegraph24k/'
    #abs_fold = '/fm1/wangziwei/wzw/TGRNet-main/datasets/tablegraph24k/'
    #abs_fold = '/fm1/wangziwei/wzw/pubtabnet/pubtabnet_TGRNet/'
    # abs_fold = read_files_in_folder(img_dir)
    
    # model
    # pse_inference = Detect_Inference(args.pse_config, args.pse_checkpoint, device)
    # master_inference = Recognition_Inference(args.master_config, args.master_checkpoint, device)
    # end2end = End2End(pse_inference, master_inference)
    tablemaster_inference = Structure_Recognition(args.tablemaster_config, args.tablemaster_checkpoint, device)
    
    test_demo_num = 0
    
    teds = []
    if args.data_name == 'pubtabnet':
        teds = {}
        with jsonlines.open('/data/xuyilun/project/LORE-TSR/data/pubtabnet_lore/PubTabNet_2.0.0_val.jsonl', 'r') as f:
            for tab in f:
                teds[tab['filename']] = format_html(tab)
    
    with jsonlines.open(args.ann_path, 'r') as json_f:
        for ann in tqdm(json_f):
            img_path = os.path.join(img_dir, ann['filename'])
            image = cv2.imread(img_path)
            if not os.path.exists(img_path): continue
            
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
                show_results(ann['filename'], image, preds['pred_bbox'], preds['pred_lloc'], args.out_dir)
            
            gt_bbox = get_label(ann)
            gts = {'gt_bbox': gt_bbox, 'gt_lloc': np.empty((0, 4))}
            
            if args.data_name == 'pubtabnet':
                html_code = ''.join(pred_ted_list_new)
                html_code = '''<html><body><table>%s</table></body></html>''' % html_code
                preds['pred_html'] = html_code
                gts['gt_html'] = teds[ann['filename']]
            elif args.data_name == 'sym':
                html_code = ''.join(pred_ted_list_new)
                html_code = '''<html><body><table>%s</table></body></html>''' % html_code
                preds['pred_html'] = html_code
                gts['gt_html'] = format_html(ann)
            
            eval_fn.run_one_step(preds, gts)
    
            # for img_path in tqdm(test_data):
            #     if args.data_name == 'tablegraph24k':
            #         img_name = get_img_name(img_path)[:-4] #name_org.png
            #     else:
            #         img_name = get_img_name(img_path)
            #     if not os.path.exists(abs_fold+'ann/'+img_name+'.json'): continue
                    
            #     gt_bbox, gt_logi, split = get_label(abs_path=abs_fold+'ann/', img_name=img_name)

            #     if args.data_name == 'hubtabnet' and split != 'val': continue
            #     elif split != 'test': continue
                
            #     test_demo_num += 1

            #     # OCR end2end predict
            #     # end2end = End2End(pse_inference, master_inference)
            #     # end2end_result, end2end_result_dict = end2end.predict(img_path)
            #     # torch.cuda.empty_cache()
                
            #     # table structure predict
            #     tablemaster_result, tablemaster_result_dict = tablemaster_inference.predict_single_file(img_path)
            #     torch.cuda.empty_cache()
                
            #     pred_ted_list = tablemaster_result['text'].split(',')
            #     pred_ted_list_new = []
            #     for i in range(len(pred_ted_list)):
            #         if pred_ted_list[i] == '<td></td>':
            #             pred_ted_list_new.append('<td>')
            #             pred_ted_list_new.append('</td>')
            #         else:
            #             pred_ted_list_new.append(pred_ted_list[i])
                        
            #     pred_bboxs = clear_zeros(tablemaster_result['bbox'])
            #     gts = {'gt_bbox': gt_bbox, 'gt_lloc': gt_logi}
            #     preds = {'pred_bbox': pred_bboxs, 'pred_lloc': np.empty((0, 4))}
            
            #     eval_fn.run_one_step(preds, gts)
            #     break
                
            # merge result by matcher
            # matcher = DemoMatcher(end2end_result_dict, tablemaster_result_dict)
            # match_results = matcher.match()
            # merged_results = matcher.get_merge_result(match_results)

            # # save predict result
            # for k in merged_results.keys():
            #     img = cv2.imread(img_path)
            #     bboxes = tablemaster_result_dict[k]['bbox']
            #     for box in bboxes:
            #         if box[0] == 0. and box[1] == 0. and box[2] == 0. and box[3] == 0.:
            #             continue
            #         cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 1)
            #         cv2.imwrite(f'/data/xuyilun/TM_tablegraph24k_res/{img_name}.jpg', img)
            #         cv2.imwrite(f'/data/xuyilun/icdar13_res/{img_name}.jpg', img)
            #         cv2.imwrite(f'/data/xuyilun/comfintab_res/{img_name}.jpeg', img)
                    
            #         cv2.imwrite(f'/data/xuyilun/tablegraph24k_res/{img_name}.png', img)
            #         cv2.imwrite(f'/data/xuyilun/hubtabnet_res/{img_name}.png', img)
            #         html_file_path = os.path.join(args.out_dir, k.replace('.png', '.html'))
            #         with open(html_file_path, 'w', encoding='utf-8') as f:
            #             write to html file
            #         html_context = htmlPostProcess(merged_results[k])
            #         f.write(html_context)
            
    res = eval_fn.summary_for_final_results()
    print(f'{args.data_name}: The number of test demos is {test_demo_num}.')
    print(res)
