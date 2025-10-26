# Minimal training script

import logging

import numpy as np
try:
    from pytorch3d.transforms import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
except:
    print('no pytorch3d')
import torch
from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)
import functools
import math
import io
import os
import random
import re
import pickle
from multiprocessing import Value
from functools import partial
import json
from itertools import chain
from dataclasses import dataclass
import numpy as np
from PIL import Image
import copy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from petrel_client.client import Client
except:
    pass 
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
from PIL import Image
import clip
from pdb import set_trace
import h5py
from scipy.spatial.transform import Rotation as R
import time

from utils.argument_utils import get_parser

from data.libero_seer import get_libero_pretrain_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
# Minimal setup to build the data loader
clip_model, image_processor = clip.load("ViT-B/32", device=device)
parser = get_parser(is_eval=False)
args = parser.parse_args()
# print(args)

dataset = get_libero_pretrain_dataset(args, image_processor, clip, epoch=0, floor=False)
for data in dataset.dataloader:
    breakpoint()
breakpoint()