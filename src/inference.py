
import torch

import os
import argparse
from torch.utils.data import DataLoader

import torch
import pickle
from utils.params import Params
# from torch_geometric.transforms import NormalizeFeatures
import mydataset
from LMs.myinferencer import MyInferencer
import LMs
from utils import util


def parse_args(config_path):
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = config_path)
    return parser.parse_args()

if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args('config/inference.ini') # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # params.device = torch.device('cpu')
    print('Using device:', params.device)

    # section 2, get the model
    model = LMs.setup(params).to(params.device)

    #section 3, trainer
    # mytrainer = MyTrainer(params)

    # section 3,data
    # this is usually covered by huggingface models
    # params.output_dir = 'tmp_dir/'
    # for file_path in [
    #     '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_a1_dataset.hf',
    #     # '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_b1_dataset.hf',
    #     # '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_b2_dataset.hf',
    #     # '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_b3_dataset.hf',
    # ]:
    #     print('-- prepare:', file_path)
    #     params.cdip_path = file_path
    # print('-- load raw:', params.cdip_path)
    mydata = mydataset.setup(params)
    # print('-- finished mapping, now inference:', params.cdip_path)

    myinferencer = MyInferencer(params)

    # section 6, classifying and decoding labels
    # img_paths,all_preds = myinferencer.inference_for_classification(params, model, mydata)
    # print('finished infering, and prepare to write:',len(img_paths))
    # for img, pred in zip(img_paths,all_preds):
    #     label = model.config.id2label[pred]
    #     util.write_line('class_a.txt', img.strip() + '\t' + str(label))
    # print('--- end of infer for:', file_path)


    # section 7, QA infering and output data
    myinferencer.inference_for_QA(model,mydata,'docvqa_3.json')


