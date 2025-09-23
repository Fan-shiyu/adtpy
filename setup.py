from setuptools import setup, find_packages

setup(
    name='adtpy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy',
        'matplotlib-venn'
    ],
    author='Shiyu Fan',
    description='A Python package for computing similarity metrics between datasets.',
    license='MIT',
)
