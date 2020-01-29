import setuptools
import os
from io import open


here = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(name='isajosep_util',
                 version='0.5.2',
                 description="Misc utilities",                 
                 author="Isaac Joseph",
                 classifiers=['Programming Language :: Python :: 3.6',
                              'Programming Language :: Python :: 2.7'],
                 install_requires=['pandas'],
                 packages=['isajosep_util'],  python_requires='>= 2.7')
