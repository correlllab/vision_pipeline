# setup.py
from setuptools import setup, find_packages
import io, os

# read your README for the long description
here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='vision-pipeline',
    version='0.1.0',
    author='Your Name',
    author_email='you@example.com',
    description='Probabilistic 3D vision pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/VisionPipeline',
    packages=find_packages(exclude=['tests', '__pycache__']),
    include_package_data=True,            # pick up config.json + figures
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'opencv-python',
        'open3d',
        # …any other deps…
    ],
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
