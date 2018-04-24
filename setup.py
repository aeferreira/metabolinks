# -*- coding: utf8 -*-

import codecs
import os
import re

from setuptools import find_packages, setup


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

###############################################################################



LONG = (
    read("README.rst") + "\n\n" +
    "Release Information\n" +
    "===================\n\n" +
    re.search("(\d+.\d.\d \(.*?\)\n.*?)\n\n\n----\n\n\n",
              read("CHANGELOG.rst"), re.S).group(1) +
    "\n\n`Full changelog " +
    "<{uri}en/stable/changelog.html>`_.\n\n".format(uri=URI) +
    read("AUTHORS.rst")
)


if __name__ == "__main__":
    setup(
        name="metabolinks",
        license="License :: OSI Approved :: MIT License",
        url='https://github.com/aeferreira/metabolinks',
        download_url='https://github.com/aeferreira/metabolinks',
        version='0.51',
        author='António Ferreira and Gil Pires',
        author_email="aeferreira@fc.ul.pt",
        maintainer='António Ferreira',
        maintainer_email="aeferreira@fc.ul.pt",
        classifiers=CLASSIFIERS,
        keywords=keywords=['Metabolomics', 'Mass Spectrometry',
                  'Data Analysis', 'Ultra-high resolution MS'],
        description="A set of tools for high-resolution MS metabolomics data analysis",
        long_description="""A set of tools for high-resolution MS metabolomics data analysis.
        
        Metabolinks aims at providing several tools that streamline most of
        the metabolomics workflow. These tools were written having ultra-high
        resolution MS based metabolomics in mind.
        Features are a bit scarce right now:
        
        - peak list alignment
        - data matrix filtering, convertion and sample similarity
        - compound taxonomy retrieval
        
        But our road map is clear and we expect to stabilize in a beta version pretty soon.
        Stay tuned, and check out examples (jupyter notebooks).
        """,
        packages=['metabolinks'],
        zip_safe=False,
        install_requires=['six', 'requests', 'numpy', 'pandas', 'pytest', 'matplotlib>=2.0'],
    )
        
