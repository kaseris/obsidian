import pytest


def test_imports():
    import obsidian
    import obsidian.blocks
    import obsidian.coco
    import obsidian.core
    import obsidian.blocks.classification
    import obsidian.blocks.detection
    import obsidian.blocks.losses
    import obsidian.blocks.pooling
    import obsidian.datasets
    import obsidian.datasets.dataset
    import obsidian.datasets.transforms
    assert True


def test_import_pycocotools():
    import pycocotools
    assert True
