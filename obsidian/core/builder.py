import pprint
import obsidian.module as om
import obsidian.blocks as blocks
from obsidian.core.fileclient import read_file


def _build_pointcloud_classifier(model_cfg):
    pass


def _build_image_classifier(model_cfg):
    pass


def _build_image_detector(model_cfg):
    registry = blocks.DETECTORS
    print(model_cfg.name, model_cfg.args)
    model = registry[model_cfg.name](**model_cfg.args)
    model = registry['FashionDetector'](**{'detector': model})
    return model


task2builder = {
    'pointcloud_classification': _build_pointcloud_classifier,
    'image_classification': _build_image_classifier,
    'image_detection': _build_image_detector
}


class Config:
    def __init__(self, module_cfg) -> None:
        self.name = module_cfg['name']
        self.args = module_cfg['cfg']

    def build_from_config(self):
        pass


class ConfigBuilder:
    def __init__(self, cfg_filename: str) -> None:
        self.cfg_filename = cfg_filename
        self.configuration = read_file(cfg_filename)
        # TODO: Must perform a structure check of the configuration.
        # Can't afford to throw in garbage.
        self._configs = self.configuration.keys()
        self.build_fn = None
        self.build_componenents()

    def build_componenents(self):
        task = self.configuration['task']
        self.build_fn = task2builder[task]
        model = self.build_fn(Config(self.configuration['model']))
        setattr(self, 'model', model)

    @property
    def configs(self):
        return self._configs

    def __repr__(self):
        config_str = "\n"
        for key, value in self.configuration.items():
            config_str += f"    {key}: "
            if isinstance(value, dict):
                config_str += "\n"
                for subkey, subvalue in value.items():
                    config_str += f"        {subkey}: {subvalue},\n"
                config_str += "    ,\n"
            else:
                config_str += f"{value},\n"
        config_str += "\n"
        config_str += f"build_fn:  {self.build_fn}"
        return config_str


if __name__ == '__main__':
    config_builder = ConfigBuilder('configs/detection/base.yaml')
    # print(config_builder)
    print(config_builder.model)
    print(isinstance(config_builder.model, om.OBSModule))
