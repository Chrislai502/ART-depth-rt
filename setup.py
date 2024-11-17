# setup.py
from setuptools import setup, find_packages

setup(
    name='artdepth',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            # Uncomment and customize if you want to add command-line tools
            # 'artdepth-train=art_depth.train:main', 
        ],
    },
    author='Chris Lai',
    author_email='chrislai_502@berkeley.edu',
    description='A brief description of the artdepth package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Chrislai502/ART-depth-rt',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10.14',
)
