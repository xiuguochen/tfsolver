from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

setup(
    name='tfsolver',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    description='A numerical Python toolkit for parallel electromagnetic calculation of planar multilayer thin films at multi-wavelength and multi-angle',
    author='xiuguochen',
    author_email='xiuguochen@hust.edu.cn',
    url='https://github.com/xiuguochen/tfsolver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['optics', 'multilayer thin film', 'python'],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Intended Audience :: Science/Research'
    ]
)