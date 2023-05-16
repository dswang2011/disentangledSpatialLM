
import torch

import os
import argparse
from torch.utils.data import DataLoader

import torch
import pickle
from utils.params import Params
# from torch_geometric.transforms import NormalizeFeatures
# import pretrain_dataset
from LMs.HFTrainer import MyTrainer
import LMs
import mydataset


def parse_args(config_path):
    parser = argparse.ArgumentParser(description='pretrain the model')
    parser.add_argument('--config', dest='config_file', default = config_path)
    return parser.parse_args()
    
if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args('config/pretrain.ini') # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', params.device)

    # section 2, objective function and output dim/ move to trainer
    # this is usually covered by huggingface models

    # section 3, model, loss function, and optimizer
    # load from cs model checkpoint
    model = LMs.setup(params).to(params.device)
    # model = LMs.setup(params)

    # confirm: to be saved path:
    print(params.checkpoint_save_path)
    print(params.save_path)
    
    # section 4, data and saving path for output model
    mydata = mydataset.setup(params)

    # # section 5, pretrain
    mytrainer = MyTrainer(params)
    mytrainer.pretrain(params, model, mydata)


