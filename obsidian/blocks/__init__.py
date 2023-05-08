from .pointcloud import stn as stn
from .pointcloud import pointnet as pointnet
from .pooling import pooling as pooling
from .detection import detection as detection

from .pointcloud.pointnet import PointNetfeat as PointNetfeat
from .pointcloud.pointnet import PointNetDenseCls as PointNetDenseCls
from .pointcloud.pointnet import PointNetCls as PointNetCls
from .pointcloud.stn import STNkd as STNkd
from .pointcloud.stn import STN3d as STN3d

from .classification import model as model
from .classification.registry_ import MODELS as MODELS
from .classification.registry_ import BACKBONES as BACKBONES
from .classification.registry_ import CLS_HEADS as CLS_HEADS
from .classification.model import resnet_18 as resnet_18
from .classification.model import resnet_50 as resnet_50
from .classification.model import resnet_152 as resnet_152
from .classification.model import vit_b_16 as vit_b_16
from .classification.model import ClassificationHead as ClassificationHead
from .classification.model import LinearClassificationHead as LinearClassificationHead
from .classification.model import CombinedGlobalDescriptorClassHead as CombinedGlobalDescriptorClassHead
from .classification.model import ResNetDeepFashion as ResNetDeepFashion
from .classification.model import FashionDetector as FashionDetector

from .detection.registry_ import DETECTORS as DETECTORS
from .detection.registry_ import DETECTOR_WEIGHTS as DETECTOR_WEIGHTS
from .detection.registry_ import BACKBONES as DETECTOR_BACKBONES
from .detection.registry_ import BACKBONE_WEIGHTS as DETECTOR_BACKBONE_WEIGHTS

from .pooling.registry_ import POOLING as POOLING
from .pooling.registry_ import DESCRIPTORS as DESCRIPTORS
from .pooling.pooling import LandmarkPooling as LandmarkPooling
from .pooling.pooling import GlobalPooling as GlobalPooling
from .pooling.pooling import RoIPooling as RoIPooling
from .pooling.pooling import GlobalDescriptor as GlobalDescriptor
from .pooling.pooling import CombinedGlobalDescriptor as CombinedGlobalDescriptor
