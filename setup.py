import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='obsidian',
    author='Michalis Kaseris',
    version='0.1',
    install_requires=required,
    packages=find_packages()
)

# Change the current working directory to obsidian/coco/PythonAPI
os.chdir('obsidian/coco/PythonAPI')

# Execute the 'make' command to build the C++ components
os.system('make')

# Execute the 'python setup.py install' command to install the package
os.system('python setup.py install')

os.chdir('.')
# Execute pytest
os.system('pytest')
