import logging
from obsidian.core import builder, read_file

logging.basicConfig(level=logging.DEBUG)

config = 'configs/detection/base.yaml'

b = builder.ConfigBuilder(config)

print(dir(b))
