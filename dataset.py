import os
import os.path as osp
from os import PathLike

import torch
from torch.utils.data import Dataset

from typing import Union

import config
from registry import Registry

DATASETS = Registry()


@DATASETS.register('deepfashion_cat_att')
class DeepFashionCategoryAttribute(Dataset):
    """
    PyTorch Dataset class for the DeepFashion dataset for category and attribute classification task.

    Args:
        include_attributes (bool): Whether to include attribute labels in the dataset (default: False).

    Attributes:
        data_dir (str): Path to the DeepFashion dataset directory.
        CLASS_LABELS (List[str]): List of class labels.
        ATTR_LABELS (List[str]): List of attribute labels.
        IDX_TO_CLASS (Dict[int, str]): Dictionary mapping class index to class label.
        CLS_TO_IDX (Dict[str, int]): Dictionary mapping class label to class index.
        DATA_DICT (Dict[int, Dict[str, Union[str, int, List[int]]]]): Dictionary containing information
            about each image in the dataset. The keys are the indices and the values are another dictionary
            containing the keys "path", "cat_index", "category", and "attributes" (if include_attributes is True).

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(index: int) -> Dict[str, Union[str, int, List[int]]]: Returns a dictionary containing
            information about the image at the given index. The dictionary contains the keys "path", "cat_index",
            "category", and "attributes" (if include_attributes is True).

    """

    def __init__(self,
                 include_attributes=False):
        self.data_dir = config.DEEP_FASHION_DIR
        self.CLASS_LABELS = []
        self.ATTR_LABELS = []

        with open(config.DEEP_FASHION_CLOTHING_CATEGORIES_PATH, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 1:
                    self.CLASS_LABELS.append(line.split()[0])

        with open(config.DEEP_FASHION_CLOTHING_ATTRIBUTES_PATH, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 1:
                    self.ATTR_LABELS.append(
                        ' '.join(l for l in line.split() if l.replace('-', '').isalpha()))

        self.IDX_TO_CLASS = {idx + 1: c for idx, c in enumerate(self.CLASS_LABELS)}


        self.CLS_TO_IDX = {v: k for k, v in self.IDX_TO_CLASS.items()}

        self.DATA_DICT = dict()

        with open(config.DEEP_FASHION_CLOTHING_LIST_CAT_IMG_PATH, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 1:
                    self.DATA_DICT[idx-2] = {
                        "path": line.split()[0],
                        "cat_index": int(line.split()[1]),
                        "category": self.IDX_TO_CLASS[int(line.split()[1])]
                    }
        if include_attributes:
            with open(config.DEEP_FASHION_CLOTHING_LIST_ATT_IMG_PATH, 'r') as f:
                for idx, line in enumerate(f):
                    if idx > 1:
                        self.DATA_DICT[idx-2]['attributes'] = [idx for idx,
                                                        att in enumerate(line.split()[1:]) if int(att) > 0]

    def __len__(self):
        return len(list(self.DATA_DICT.keys()))

    def __getitem__(self, index):
        return self.DATA_DICT[index]
