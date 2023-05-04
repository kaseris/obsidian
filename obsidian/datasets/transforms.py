import torch
import torchvision.transforms as transforms

from obsidian.coco import transforms as T
from obsidian.core.registry import Registry

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
    - Resizing the image to the size specified in the `obsidian.core.config` module.
    - Randomly cropping a portion of the image to the size specified in the `obsidian.core.config` module.
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
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


@TRANSFORMS.register('DeepFashion_validation_tf')
def deepfashion_validation_transform():
    """
    Returns a set of image transforms commonly used for validating DeepFashion datasets.

    The transforms include:
    - Resizing the image to the size specified in the `obsidian.core.config` module.
    - Randomly cropping a portion of the image to the size specified in the `obsidian.core.config` module.
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
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
