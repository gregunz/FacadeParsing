#!/usr/bin/env python

from distutils.core import setup

setup(name='FacadeParsing',
      version='1.0.1',
      description='Facade parsing project',
      author='Grégoire Clément',
      author_email='gregoire.clement@epfl.ch',
      url='github.com/gregunz',
      packages=['facade_project'],
      requires=[
          'tqdm',
          'torch',
          'pillow',
          'numpy',
          'labelme',
          'matplotlib',
          'torchvision',
          'imageio',
          'shapely'
      ])
