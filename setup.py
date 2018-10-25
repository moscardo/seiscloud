import sys

from setuptools import setup, find_packages


version = '0.1'


setup(
    version=version,
    author='Simone Cesca',
    author_email='simone.cesca@gfz-potsdam.de',
    license='GPLv3',
    name='seiscloud',
    description='Clustering seismicity again?',
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, <4',
    install_requires=[],
    packages=['seiscloud'],
    package_dir={'seiscloud': 'src'},
    scripts=['src/seiscloud'],
)
