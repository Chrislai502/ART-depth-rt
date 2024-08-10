from setuptools import setup, find_packages

setup(
    name='vision-ik',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        'numpy',
        'torch',
        # other dependencies
    ],
    entry_points={
        'console_scripts': [
            # If you want to create command-line tools
            # 'vision-ik-train=vision_ik.train:main', # Assuming `train.py` has a main function
        ],
    },
    author='Chris Lai',
    author_email='cl@co.bot',
    description='A brief description of Vision-ik package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/vision-ik',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
