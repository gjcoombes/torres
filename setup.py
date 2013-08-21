# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='torres',
    version='0.0.1',
    description='Manipulation and presentation of shapefiles and maps',
    long_description=readme,
    author='Gavin Coombes',
    author_email='gcoombes@apasa.com.au',
    url='https://github.com/gjcoombes/torres.git',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

