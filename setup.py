from setuptools import setup
from setuptools.command.install import install

version = '0.1'


# prevent setuptools from installing as egg
class CustomInstallCommand(install):
    def run(self):
        install.run(self)


setup(
    cmdclass={
        'install': CustomInstallCommand,
    },
    version=version,
    author='Simone Cesca',
    author_email='simone.cesca@gfz-potsdam.de',
    license='GPLv3',
    name='seiscloud',
    description='Clustering seismicity again?',
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, <4',
    install_requires=[],
    packages=[
        'seiscloud',
        'seiscloud.apps'],
    package_dir={'seiscloud': 'src'},
    entry_points={
        'console_scripts': [
            'seiscloud = seiscloud.apps.seiscloud:main',
        ]
    },
)
