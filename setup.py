from setuptools import setup, find_packages

setup(
    name='PCHSegmentation',
    version='0.0.3',
    description='A Python package for PCA-based segmentation of 3D point clouds',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Alex Sumozas',
    author_email='your.email@example.com',
    url='meter my git',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],
    extras_require={
        'test': ['matplotlib'],
        '3d': ['open3d'],
        'all': ['open3d', 'matplotlib']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
