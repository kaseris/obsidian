import obsidian
from obsidian.core.builder import ConfigBuilder


def test_model_builder():
    builder = ConfigBuilder(cfg_filename='configs/detection/base.yaml')
    assert isinstance(builder.model, obsidian.module.OBSModule)
