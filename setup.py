# -*- coding: utf-8 -*-
"""`metabolinks` is on `Github`_.

.. _github: https://github.com/aeferreira/metabolinks

"""
from io import open
from setuptools import setup, find_packages

def read_file(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as f:
        content = f.read()
    return content

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
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Information Analysis"
]

if __name__ == "__main__":
    setup(
        name="metabolinks",
        license="MIT",
        url='https://github.com/aeferreira/metabolinks',
        download_url='https://github.com/aeferreira/metabolinks',
        version='0.51',
        zip_safe=False,
        author='António Ferreira and Gil Pires',
        author_email="aeferreira@fc.ul.pt and gilpires071997@gmail.com",
        maintainer='António Ferreira',
        maintainer_email="aeferreira@fc.ul.pt",
        classifiers=CLASSIFIERS,
        keywords=['Metabolomics', 'Mass Spectrometry',
                  'Data Analysis', 'Ultra-high resolution MS'],
        description="A set of tools for high-resolution MS metabolomics data analysis",
        long_description=read_file('README.rst'),
        packages=['metabolinks'],
        package_data={'metabolinks': [
            'examples/peak_alignment_xcel.ipynb',
            'examples/taxonomy_annotation_example.ipynb',
            'data/data_to_align.xlsx',
            'data/MassTRIX_output.tsv'
        ]},
        include_package_data=True,
        install_requires=['six', 'requests', 'numpy', 
                          'pandas', 'xlrd', 'xlsxwriter',
                          'pytest', 'matplotlib>=2.0'],
    )
        
