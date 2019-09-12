from setuptools import setup
from setuptools import find_packages

setup(name='DDrop',
      version='0.0.8',
      description='Mathematical Dropconnect',
      author='Derek Khu, Andrea Santilli',
      author_email='derek.khu@gmail.com',
      license='MIT',
      install_requires=['keras'],
      packages=find_packages())
