import sys
import os
import os.path as osp
import numpy as np
import math
import random
import time
import logging
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split, Subset
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DataParallel
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import statistics

from opt import parse_opts
from models import get_model
from dataloader.multiloader import MyData
from utils.ckpt_util import save_ckpt
from utils.cache_data import have_cached_data, cache_data, get_cached_data

import inspect
import argparse
import json

with open("args.json", "r") as json_file:
    args_dict = json.load(json_file)

args = argparse.Namespace(**args_dict)

dataset = MyData(
    args.raw_mrna_path.format(args.cancer_type),
    args.raw_cnv_path.format(args.cancer_type),
    args.raw_methylation_path.format(args.cancer_type),
    args.node_path,
    args.edge_path.format(args.cancer_type),
    args.kegg_path,
    args.clinical_path.format(args.cancer_type),
    args,
)

# with open('test.txt', 'w') as f:
#     f.write(a)

args.node_num = dataset.get_node_num()
args.omics_num = len(dataset.omics_types)
labels = dataset.get_labels()


