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
    dataset = DeepFashionCategoryAttribute(include_attributes=include_attributes, transform=transform)
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
    A PyTorch dataset for DeepFashion category and attribute classification.

    Args:
        data_dir (str, optional): The directory containing the DeepFashion dataset. Defaults to `config.DEEP_FASHION_DIR`.
        include_attributes (bool, optional): Whether to include attribute labels in the output. Defaults to `False`.
        transforms (callable, optional): A function or callable object that applies transforms to input images. Defaults to `TRANSFORMS['DeepFashion_default']()`.
        split_type (str, optional): The type of split to use (`'train'`, `'val'`, or `'test'`). Defaults to `'train'`.
        split_info (dict, optional): A dictionary containing information about the dataset split. Defaults to `None`.
        garment_annotations (dict, optional): A dictionary containing annotations for each image in the dataset. Defaults to `None`.

    Attributes:
        classes (list): A list of class labels for the dataset.
        n_classes (int): The number of classes in the dataset.

    Methods:
        __init__(self, data_dir=config.DEEP_FASHION_DIR, include_attributes=False, transforms=None, split_type='train', split_info=None, garment_annotations=None):
            Initializes a new `DeepFashionCategoryAttribute` dataset.
            
        __len__(self):
            Returns the number of items in the dataset.
            
        __getitem__(self, index) -> Dict[torch.TensorType, torch.TensorType]:
            Returns the item at the specified index.
            
        classes(self):
            Returns a list of class labels for the dataset.
            
        n_classes(self):
            Returns the number of classes in the dataset.
    Example:
    ```
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
    from dataset import DeepFashionCategoryAttribute
    
    from utils import prepare_data

    # Define the transforms to be applied to each image
    transforms = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor()
    ])
    splits, garment_annotations = prepare_data()
    # Load the dataset
    dataset = DeepFashionCategoryAttribute(
        data_dir='path/to/dataset',
        include_attributes=True,
        transforms=transforms,
        split_type='train',
        split_info=splits,
        garment_annotations=garment_annotations
    )

    # Create a DataLoader to handle batching and shuffling of the data
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # Iterate over the data
    for batch in dataloader:
        images = batch['img']
        categories = batch['category']
        attributes = batch['attributes']
    ```
    """
    def __init__(self,
                 data_dir=config.DEEP_FASHION_DIR,
                 include_attributes=False,
                 transforms=None,
                 split_type='train',
                 split_info=None,
                 garment_annotations=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.include_attributes = include_attributes
        self.split_type = split_type
        
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
        
        self.split = split_info[self.split_type]
        self.annos = garment_annotations

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index) -> Dict[torch.TensorType,
                                         torch.TensorType]:
        pth = self.split[index]
        try:
            img = Image.open(osp.join(self.data_dir, pth)).convert('RGB')
        except FileNotFoundError:
            img = Image.open(osp.join(self.data_dir, pth.replace('A-Line_Dress', 'A-line_Dress'))).convert('RGB')
        crop = self.annos[pth]['bbox']
        img_cropped = img.crop(crop)
        img_transformed = self.transforms(img_cropped)
        cat = self.annos[pth]['category']
        return_dict = {'img': img_transformed,
                       'category': torch.tensor([cat - 1])}
        if self.include_attributes:
            attributes = self.annos[pth]['attributes']
            attributes = torch.clamp(torch.tensor(attributes), 0, 1)
            return_dict['attributes'] = attributes        
        return return_dict
    
    @property
    def classes(self):
        return self.CLASS_LABELS
    
    @property
    def n_classes(self):
        return len(self.CLASS_LABELS)
    