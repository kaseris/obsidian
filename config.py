import os.path as osp

# Modify the empty string so that it contains your own GCS service account key.
GCS_SERVICE_ACCOUNT_KEY = ''

DEEP_FASHION_LOW_RES_URL = 'https://storage.googleapis.com/deep-fashion-bucket/img.zip'
DEEP_FASHION_HI_RES_URL = None

DEEP_FASHION_N_CLASSES = 50

DEEP_FASHION_DIR = osp.join(osp.expanduser('~'), 'Documents', 'dev', 'DeepFashion')
DEEP_FASHION_CLOTHING_ANNOS_DIR = osp.join(DEEP_FASHION_DIR, 'Anno_coarse')
DEEP_FASHION_CLOTHING_IMAGES_DIR = osp.join(DEEP_FASHION_DIR, 'img')
DEEP_FASHION_CLOTHING_CATEGORIES_PATH = osp.join(DEEP_FASHION_CLOTHING_ANNOS_DIR, 'list_category_cloth.txt')
DEEP_FASHION_CLOTHING_ATTRIBUTES_PATH = osp.join(DEEP_FASHION_CLOTHING_ANNOS_DIR, 'list_attr_cloth.txt')
DEEP_FASHION_CLOTHING_LIST_CAT_IMG_PATH = osp.join(DEEP_FASHION_CLOTHING_ANNOS_DIR, 'list_category_img.txt')
DEEP_FASHION_CLOTHING_LIST_ATT_IMG_PATH = osp.join(DEEP_FASHION_CLOTHING_ANNOS_DIR, 'list_attr_img.txt')
