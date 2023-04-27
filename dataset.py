import os
import json
import os.path as osp

from typing import Dict

import numpy as np

import nltk
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

import config

import coco.transforms as T
from registry import Registry
from utils import prepare_data

DATASETS = Registry()
TRANSFORMS = Registry()


@TRANSFORMS.register('DeepFashionObjectDetectionDefaultTransform')
def deepfashion_object_detection_default_transform(**kwargs):
    """
    Returns a set of image transforms commonly used for training DeepFashion datasets.
    """
    train = kwargs.get('train')
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(dtype=torch.float))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

@TRANSFORMS.register('DeepFashion_default_tf')
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

@TRANSFORMS.register('DeepFashion_validation_tf')
def deepfashion_validation_transform():
    """
    Returns a set of image transforms commonly used for validating DeepFashion datasets.

    The transforms include:
    - Resizing the image to the size specified in the `config` module.
    - Randomly cropping a portion of the image to the size specified in the `config` module.
    - Converting the image to a PyTorch tensor.
    - Normalizing the image using the mean and standard deviation values provided in the torchvision documentation.

    Returns:
    - A `transforms.Compose` object that can be passed to a PyTorch `DataLoader` object.

    Example:
    ```    
    transform = deepfashion_validation_transform()
    dataset = DeepFashionCategoryAttribute(include_attributes=include_attributes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    ```
    """
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.RandomCrop(config.CROP_SIZE),
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
                 **kwargs):
        self.data_dir = config.DEEP_FASHION_DIR
        self.include_attributes = kwargs['include_attributes']
        self.split_type = kwargs['split_type']
        
        self.transforms = TRANSFORMS[kwargs['transforms']]()
        
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
        if kwargs['split_info'] is not None and kwargs['garment_annotations'] is not None:
            self.split = kwargs['split_info'][self.split_type]
            self.annos = kwargs['garment_annotations']
        else:
            self.split, self.annos = prepare_data()

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
    

@DATASETS.register('fashioniq')
class FashionIQ(Dataset):
    """
    A PyTorch dataset for the FashionIQ dataset, which consists of images and captions
    describing fashion items, and includes a candidate image and a target image for each
    item. This class takes the root directory of the dataset, the name of the JSON file
    containing the data, a vocabulary object, and optional transforms as input.

    Args:
        root (str): Root directory of the FashionIQ dataset.
        data_file_name (str): Name of the JSON file containing the data.
        vocab (Vocabulary): A Vocabulary object that maps tokens to their corresponding
            integer ids.
        transform (optional, callable): A function/transform that takes in an PIL image
            and returns a transformed version.
        return_target (optional, bool): Whether to return the target image and ASIN along
            with the candidate image and captions (default True).

    Methods:
        __getitem__(self, index): Returns a single data pair, which consists of a
            target image, a candidate image, a caption, and a dictionary containing the
            target ASIN, candidate ASIN, and the original caption texts.
        __len__(self): Returns the number of items in the dataset.

    Example:
        To use this dataset, first create a vocabulary object and specify the root directory
        and the name of the data file:

        >>> vocab = Vocabulary(special_tokens=['<pad>', '<start>', '<end>', '<unk>'])
        >>> root = '/path/to/FashionIQ/dataset/'
        >>> data_file_name = 'train.json'
        >>> dataset = FashionIQ(root, data_file_name, vocab)

        To access a specific item in the dataset, use the __getitem__ method:

        >>> target_image, candidate_image, caption, metadata = dataset[0]

        To iterate over the entire dataset, use a DataLoader:

        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch_idx, (target_images, candidate_images, captions, metadata) in enumerate(dataloader):
        ...     # Do something with the batch
    """
    def __init__(self, root, data_file_name, vocab, transform=None, return_target=True):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data: index file name.
            transform: image transformer.
            vocab: pre-processed vocabulary.
        """
        self.root = root
        with open(data_file_name, 'r') as f:
            self.data = json.load(f)
        self.ids = range(len(self.data))
        self.vocab = vocab
        self.transform = transform
        self.return_target = return_target

    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        vocab = self.vocab
        id = self.ids[index]

        candidate_asin = data[id]['candidate']
        candidate_img_name = candidate_asin + '.jpg'
        candidate_image = Image.open(os.path.join(self.root, candidate_img_name)).convert('RGB')
        if self.transform is not None:
            candidate_image = self.transform(candidate_image)

        if self.return_target:
            target_asin = data[id]['target']
            target_img_name = target_asin + '.jpg'
            target_image = Image.open(os.path.join(self.root, target_img_name)).convert('RGB')
            if self.transform is not None:
                target_image = self.transform(target_image)
        else:
            target_image = candidate_image
            target_asin = ''

        caption_texts = data[id]['captions']
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption_texts[0]).lower()) + ['<and>'] + \
                nltk.tokenize.word_tokenize(str(caption_texts[1]).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption)

        return target_image, candidate_image, caption, {'target': target_asin, 'candidate': candidate_asin, 'caption': caption_texts}

    def __len__(self):
        return len(self.ids)

@DATASETS.register('deepfashion2')
class DeepFashion2(Dataset):
    """
    A PyTorch dataset for the DeepFashion2 benchmark. The dataset implementation is similar to the
    implementation of the PennFudanPed dataset, as demonstrated in (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
    
    The `__getitem__` method should return:
    
    image: a PIL Image of size (H, W)
    target: a dict containing the following fields
        `boxes` (`FloatTensor[N, 4]`): the coordinates of the `N` bounding boxes in `[x0, y0, x1, y1]` format, ranging from `0` to `W` and `0` to `H`
        `labels` (`Int64Tensor[N]`): the label for each bounding box
        `image_id` (`Int64Tensor[1]`): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
        `area` (`Tensor[N]`): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
        `iscrowd` (`UInt8Tensor[N]`): instances with `iscrowd=True` will be ignored during evaluation.
        (optionally) `masks` (`UInt8Tensor[N, H, W]`): The segmentation masks for each one of the objects
        (optionally) `keypoints` (`FloatTensor[N, K, 3]`): For each one of the N objects, it contains the K keypoints in `[x, y, visibility]` format, defining the object. `visibility=0` means that the keypoint is not visible. Note that for data augmentation, the notion of flipping a keypoint is dependent on the data representation, and you should probably adapt references/detection/transforms.py for your new keypoint representation


    Args:
        `root (str)`: The root directory of the dataset.
        `split (str)`: The dataset split to use. It can be 'train', 'val', or 'test'.
        `transforms` (callable, optional): A function/transform that takes in an PIL image and a
            target dict, and returns a transformed version of them. Default: None.

    Attributes:
`        `root` (`str`): The root directory of the dataset.
`        `split` (`str`): The dataset split being used.
        `transforms` (`callable`): The transform function being used, if any.
        `imgs` (`list`): A list of the paths to the images in the dataset split.
        `annos` (`list`): A list of the paths to the annotations in the dataset split.

    Methods:
        `__getitem__(self, idx)`: Retrieves the `idx`-th sample from the dataset.
        `_read_anno(self, anno_filename)`: Reads the annotations from the specified file.
        `_get_garment_annos(self, anno)`: Retrieves the garment annotations for each garment from the specified
            annotation dict.
        `_get_boxes(self, anno)`: Retrieves the bounding boxes from the specified annotation dict.
        `_create_mask(self, anno, img_size)`: Creates a mask image from the specified annotation dict
            and image size.
        `__len__(self)`: Returns the number of samples in the dataset split.
    """
    def __init__(self, root, split, transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        
        imgs_path = osp.join(self.root, self.split, 'image')
        annos_path = osp.join(self.root, self.split, 'annos')
        
        self.imgs = sorted(list(map(lambda x: os.path.join(imgs_path, x),
                                    os.listdir(os.path.join(self.root, self.split, 'image')))))
        self.annos = sorted(list(map(lambda x: os.path.join(annos_path, x),
                                     os.listdir(os.path.join(self.root, self.split, 'annos')))))
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        W, H = img.size
        anno = self._read_anno(self.annos[idx])
        mask = self._create_mask(anno, (W, H))
        mask_np = np.array(mask)
        obj_ids = np.unique(mask_np)
        # first id is the bg, so we remove it
        obj_ids = obj_ids[1:]
        masks = mask_np == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = torch.as_tensor(self._get_boxes(anno), dtype=torch.float32)
        labels = [anno[item].get('category_id') for item in self._get_garment_annos(anno)]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) 
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = is_crowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
        
    def _read_anno(self, anno_filename):
        with open(anno_filename, 'r') as f:
            anno = json.load(f)
        return anno
    
    def _get_garment_annos(self, anno):
        items = list(filter(lambda x: 'item' in x, anno.keys()))
        return sorted(items)
    
    def _get_boxes(self, anno):
        boxes = []
        for item in self._get_garment_annos(anno):
            box = anno[item].get('bounding_box')
            boxes.append(box)
        return boxes
        
    def _create_mask(self, anno, img_size):
        mask = Image.new('L', img_size, 0)
        for item in self._get_garment_annos(anno):
            segmentation_mask = anno[item]['segmentation']
            W, H = img_size
            cat_id = anno[item].get('category_id')
            for polygon in segmentation_mask:
                polygon = polygon = np.array(polygon).reshape(-1, 2)
                ImageDraw.Draw(mask).polygon(tuple(map(tuple, polygon)), outline=cat_id, fill=cat_id)
                mask_pil = mask
        return mask_pil            
    
    def __len__(self):
        return len(self.imgs)