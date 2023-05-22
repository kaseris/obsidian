import logging
import obsidian.module as om
import obsidian.blocks as blocks
from obsidian.core.fileclient import read_file


def _build_pointcloud_classifier(model_cfg):
    pass


def _build_image_classifier(model_cfg):
    pass


def _build_dataset(model_cfg):
    pass


def _build_optimizer(opt_cfg):
    from obsidian.blocks import optimizers
    return optimizers.get_optimizer(opt_cfg)


def _build_loader(model_cfg):
    pass


def _build_image_detector(model_cfg):
    registry = blocks.DETECTORS
    model = registry[model_cfg.name](**model_cfg.args)
    model = registry['FashionDetector'](**{'detector': model})
    return model


task2builder = {
    'pointcloud_classification': _build_pointcloud_classifier,
    'image_classification': _build_image_classifier,
    'image_detection': _build_image_detector,
    'dataset': _build_dataset,
    'optimizer': _build_optimizer
}


class Config:
    def __init__(self, module_cfg) -> None:
        self.name = module_cfg['name']
        self.args = module_cfg['cfg']

    def build_from_config(self):
        pass


class ConfigBuilder:

    __CFG_BUILDER_REQUIRED_ARGS = ['task', 'model',
                                   'dataset_train',
                                   'optimizer',
                                   'train_loader']

    def __init__(self, cfg_filename: str) -> None:
        self.cfg_filename = cfg_filename
        self.configuration = read_file(cfg_filename)
        self._check_config_keys()
        # TODO: Must perform a structure check of the configuration.
        # Can't afford to throw in garbage.
        self._configs = self.configuration.keys()
        self.build_fn = None
        self._build_components()

    def _build_components(self):
        task = self.configuration.pop('task')
        logging.debug(f'Building model for {task} task')
        self.build_fn = task2builder[task]
        model_cfg = self.configuration.pop('model')
        model = self.build_fn(Config(model_cfg))
        setattr(self, 'model', model)
        for cfg in self._configs:
            logging.info(f'Building component {cfg}')

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

    def _check_config_keys(self):
        for key in self.__CFG_BUILDER_REQUIRED_ARGS:
            if key not in self.configuration.keys():
                raise ValueError(f"Missing required key: `{key}`")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config_builder = ConfigBuilder('configs/detection/base.yaml')
