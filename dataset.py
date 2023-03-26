import os
import os.path as osp

from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config
from registry import Registry

DATASETS = Registry()
TRANSFORMS = Registry()

@TRANSFORMS.register('DeepFashion_default')
def deepfashion_default_transform():
    """
    Returns a set of image transforms commonly used for training DeepFashion datasets.

    The transforms include:
    - Resizing the image to the size specified in the `config` module.
    - Randomly cropping a portion of the image to the size specified in the `config` module.
    - Randomly flipping the image horizontally.
    - Converting the image to a PyTorch tensor.
    - Normalizing the image using the mean and standard deviation values provided in the torchvision documentation.

    Returns:
    - A `transforms.Compose` object that can be passed to a PyTorch `DataLoader` object.

    Example:
    ```
    transform = deepfashion_default_transform()
    dataset = DeepFashionDataset('path/to/data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    ```
    """
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.RandomCrop(config.CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
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
    Example:
    ```
    from dataset import DeepFashionCategoryAttribute, TRANSFORMS

    # create an instance of the dataset
    dataset = DeepFashionCategoryAttribute(include_attributes=True,
                                        transforms=TRANSFORMS['DeepFashion_default']())

    # get information about the first image in the dataset
    sample = dataset[0]

    # print the shape of the transformed image tensor and attribute tensor
    print(sample['img'].shape)
    print(sample['attributes'].shape)
    ```
    """

    def __init__(self,
                 include_attributes=False,
                 transforms=None):
        self.data_dir = config.DEEP_FASHION_DIR
        self.transforms = transforms
        self.include_attributes = include_attributes
        
        if self.transforms is None:
            self.transforms = TRANSFORMS['DeepFashion_default']()
        
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
        with open(config.DEEP_FASHION_CLOTHING_LIST_BBOX_IMG_PATH, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 1:
                    self.DATA_DICT[idx-2]['bbox'] = line.strip().split(' ')[-4:]
        
        if include_attributes:
            with open(config.DEEP_FASHION_CLOTHING_LIST_ATT_IMG_PATH, 'r') as f:
                for idx, line in enumerate(f):
                    if idx > 1:
                        self.DATA_DICT[idx-2]['attributes'] = line.strip().split(' ')[-1000:]

    def __len__(self):
        return len(list(self.DATA_DICT.keys()))

    def __getitem__(self, index) -> Dict[torch.TensorType,
                                         torch.TensorType]:
        img = Image.open(osp.join(self.data_dir, self.DATA_DICT[index]['path'])).convert('RGB')
        crop = list(map(int, self.DATA_DICT[index]['bbox']))
        img_cropped = img.crop(crop)
        img_transformed = self.transforms(img_cropped)
        return_dict = {'img': img_transformed,
                       'category': F.one_hot(torch.tensor(self.DATA_DICT[index]['cat_index'] - 1), num_classes=len(self.CLASS_LABELS))}
        if self.include_attributes:
            attributes = list(map(int, [el if el != '' else '-1' for el in self.DATA_DICT[index]['attributes']]))
            attributes = torch.clamp(torch.tensor(attributes), 0, 1)
            return_dict['attributes'] = attributes        
        return return_dict
    