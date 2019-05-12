#!/usr/bin/env python

from distutils.core import setup

setup(name='FacadeParsing',
      version='1.0.1',
      description='Facade parsing project',
      author='Gregunz',
      author_email='mail@gregunz.io',
      url='github.com/gregunz',
      packages=['facade_project'], requires=['tqdm', 'torch', 'pillow', 'numpy', 'labelme', 'matplotlib', 'torchvision']
      )