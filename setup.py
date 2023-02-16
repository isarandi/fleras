from setuptools import setup
import os

try:
    dependencies_managed_by_conda = os.environ['DEPENDENCIES_MANAGED_BY_CONDA'] == '1'
except KeyError:
    dependencies_managed_by_conda = False

setup(
    name='fleras',
    version='0.1.0',
    author='István Sárándi',
    author_email='sarandi@vision.rwth-aachen.de',
    packages=['fleras', 'fleras.optimizers', 'fleras.layers'],
    scripts=[],
    license='LICENSE',
    description='Fleras makes using Keras more flexible!',
    long_description='Fleras provides a ModelTrainer class to help you define more complicated '
                     'forward passes, losses and metrics that need access to more things than '
                     'Keras allows. There are also a bunch of other extensions for Keras and '
                     'TensorFlow here.',
    python_requires='>=3.6',
    install_requires=[] if dependencies_managed_by_conda else [
        'tensorflow',
        'attrdict',
        'numpy'
    ]
)
