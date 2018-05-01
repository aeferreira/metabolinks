
***********
Metabolinks
***********

``Metabolinks`` is a Python package that provides a set of tools for high-resolution
MS metabolomics data analysis.
        
Metabolinks aims at providing several tools that streamline most of
the metabolomics workflow. These tools were written having ultra-high
resolution MS based metabolomics in mind.

Features are a bit scarce right now:

- peak list alignment
- data matrix filtering, conversion and sample similarity measures
- compound taxonomy retrieval

But our road map is clear and we expect to stabilize in a beta version pretty soon.

Stay tuned, and check out the examples folder (examples are provided as
jupyter notebooks).

Installing
==========

``Metabolinks`` is distributed on PyPI_ and can be installed with pip on
a Python 3.4+ installation::

   pip install metabolinks

.. _PyPI: https://pypi.org/project/metabolinks


However, even if ``Metabolinks`` is written in Python, it requires some of the powerful scientific
packages that are pre-installed on "Scientific/Data Science Python" distributions.

One of these two products is highly recommended:

- `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ (or `Miniconda <http://conda.pydata.org/miniconda.html>`_ followed by the necessary ``conda install``'s)
- `Enthought Canopy <https://www.enthought.com/products/canopy/>`_

The formal requirements are:

- Python 3.4 and above (it runs on Python 2.7 too)
- ``setuptools``, ``pip``, ``six``, ``requests`` and ``pytest``
- ``numpy``, ``matplotlib`` and ``pandas``

The installation of the ``Jupyter`` platform is also recommended since
the examples are provided as *Jupyter notebooks*.

