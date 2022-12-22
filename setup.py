"""`metabolinks` is on `Github`_.

.. _github: https://github.com/aeferreira/metabolinks

"""
from io import open

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def read_file(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as f:
        return f.read()

###############################################################################

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Education',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Information Analysis"
]

requires = ['requests', 'numpy', 
            'pandas>=0.25', 'xlrd', 'xlsxwriter', 'pandas-flavor', 'six',
            'pytest', 'matplotlib>=2.0']

packages = ['metabolinks', 'tests', 'notebooks']

setup(
    name="metabolinks",
    license="MIT",
    url='https://github.com/aeferreira/metabolinks',
    download_url='https://github.com/aeferreira/metabolinks',
    version='0.75',
    zip_safe=False,
    author='António Ferreira',
    author_email="aeferreira@fc.ul.pt",
    maintainer='António Ferreira',
    maintainer_email="aeferreira@fc.ul.pt",
    classifiers=CLASSIFIERS,
    keywords=['Metabolomics', 'Mass Spectrometry', 'Data Analysis', 'Ultra-high resolution MS'],
    description="A set of tools for high-resolution MS metabolomics data analysis",
    long_description=read_file('README.rst'),
    long_description_content_type='text/x-rst',
    packages=packages,
    include_package_data=True,
    package_data={'metabolinks': ['data/sample_data.xlsx'],
                  "notebooks": ["blacklist.txt", "MassTRIX_output.tsv", '*.ipynb', '*.xlsx']},
    install_requires=requires
)

