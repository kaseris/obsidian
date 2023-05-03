import importlib
import os.path as osp

from os import PathLike
from typing import Callable, Union

import config


def import_registry(module_name: str, registry_name):
    module = importlib.import_module(module_name)
    registry = getattr(module, registry_name)
    return registry


def read_file_helper(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with open(filename, 'r') as file:
                data = file.readlines()
            return func(data[2:], *args, **kwargs)
        return wrapper
    return decorator


@read_file_helper(config.DEEP_FASHION_SPLITS_PATH)
def prepare_splits(data, *args, **kwargs) -> dict[str, list[str]]:
    """
    Parse the DeepFashion splits file and return a dictionary of image paths for each split.

    Args:
        data: A list of strings, where each string represents a line in the DeepFashion splits file.
        *args: Additional positional arguments to pass to the function (unused).
        **kwargs: Additional keyword arguments to pass to the function (unused).

    Returns:
        A dictionary of image paths for each split, with the following keys:
        - 'train': A list of image paths for the training split.
        - 'val': A list of image paths for the validation split.
        - 'test': A list of image paths for the testing split.

    Raises:
        None.
    """
    splits = {'train': [],
              'val': [],
              'test': []}
    for ln in data:
        _split = ln.strip().split(' ')[-1]
        _img_path = ln.strip().split(' ')[0].replace(
            'A-line_Dress', 'A-Line_Dress')
        splits[_split].append(_img_path)
    return splits


@read_file_helper(config.DEEP_FASHION_CLOTHING_LIST_CAT_IMG_PATH)
def prepare_categories(data, *args, **kwargs):
    """
    Parse the DeepFashion clothing categories file and return a dictionary of annotations for each image.

    Args:
        data: A list of strings, where each string represents a line in the DeepFashion clothing categories file.
        *args: Additional positional arguments to pass to the function. The first argument should be a dictionary of annotations for each image.
        **kwargs: Additional keyword arguments to pass to the function (unused).

    Returns:
        A dictionary of annotations for each image, where each key is an image path and each value is a dictionary with the following keys:
        - 'category': An integer representing the clothing category.

    Raises:
        None.
    """
    annotations = args[0]
    for ln in data:
        _path = ln.strip().split(' ')[0].replace(
            'A-line_Dress', 'A-Line_Dress')
        _cat = ln.strip().split(' ')[-1]
        annotations[_path] = {'category': int(_cat)}
    return annotations


@read_file_helper(config.DEEP_FASHION_CLOTHING_LIST_BBOX_IMG_PATH)
def prepare_bboxes(data, *args, **kwargs):
    """
    Parse the DeepFashion clothing bounding boxes file and update the dictionary of annotations for each image.

    Args:
        data: A list of strings, where each string represents a line in the DeepFashion clothing bounding boxes file.
        *args: Additional positional arguments to pass to the function. The first argument should be a dictionary of annotations for each image.
        **kwargs: Additional keyword arguments to pass to the function (unused).

    Returns:
        The updated dictionary of annotations for each image, where each key is an image path and each value is a dictionary with the following keys:
        - 'category': An integer representing the clothing category.
        - 'bbox': A list of four integers representing the bounding box coordinates (left, top, width, height).

    Raises:
        None.
    """
    annotations = args[0]
    for ln in data:
        _path = ln.strip().split(' ')[0].replace(
            'A-line_Dress', 'A-Line_Dress')
        _bbox = ln.strip().split(' ')[-4:]
        _bbox = list(map(lambda x: int(x), _bbox))
        annotations[_path]['bbox'] = _bbox
    return annotations


@read_file_helper(config.DEEP_FASHION_CLOTHING_LIST_ATT_IMG_PATH)
def prepare_attributes(data, *args, **kwargs):
    """
    Parse the DeepFashion clothing attributes file and update the dictionary of annotations for each image.

    Args:
        data: A list of strings, where each string represents a line in the DeepFashion clothing attributes file.
        *args: Additional positional arguments to pass to the function. The first argument should be a dictionary of annotations for each image.
        **kwargs: Additional keyword arguments to pass to the function (unused).

    Returns:
        The updated dictionary of annotations for each image, where each key is an image path and each value is a dictionary with the following keys:
        - 'category': An integer representing the clothing category.
        - 'bbox': A list of four integers representing the bounding box coordinates (left, top, width, height).
        - 'attributes': A list of integers representing the clothing attributes.

    Raises:
        None.
    """
    annotations = args[0]
    for ln in data:
        _path = ln.strip().split(' ')[0].replace(
            'A-line_Dress', 'A-Line_Dress')
        _atts = ln.strip().split(' ')[-1000:]
        _atts = [el if el != '' else '-1' for el in _atts]
        _atts = list(map(lambda x: int(x), _atts))
        annotations[_path]['attributes'] = _atts
    return annotations


def prepare_data():
    """
    Read the DeepFashion clothing splits, categories, bounding boxes, and attributes files, and return the corresponding data.

    Args:
        None.

    Returns:
        A tuple containing two dictionaries:
        - The first dictionary contains the image paths for the training, validation, and testing sets, where each key is a split name ('train', 'val', 'test') and each value is a list of image paths.
        - The second dictionary contains the annotations for each garment image, where each key is an image path and each value is a dictionary with the following keys:
            - 'category': An integer representing the clothing category.
            - 'bbox': A list of four integers representing the bounding box coordinates (left, top, width, height).
            - 'attributes': A list of integers representing the clothing attributes.

    Raises:
        None.
    """
    splits = prepare_splits()
    garment_annos = dict()
    garment_annos = prepare_categories(garment_annos)
    garment_annos = prepare_bboxes(garment_annos)
    garment_annos = prepare_attributes(garment_annos)
    return splits, garment_annos
