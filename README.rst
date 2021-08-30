
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
- common metabolomics data-matrix preprocessing, based on ``pandas`` and ``scikit-learn``
- compound taxonomy retrieval

But our road map is clear and we expect to stabilize in a beta version pretty soon.

Stay tuned, and check out the examples folder (examples are provided as
jupyter notebooks).

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5336951.svg
   :target: https://doi.org/10.5281/zenodo.5336951

Installing
==========

``Metabolinks`` is distributed on PyPI_ and can be installed with pip on
a Python 3.6+ installation::

   pip install metabolinks

.. _PyPI: https://pypi.org/project/metabolinks


However, it is recommended to install the the scientific Python packages that are
required by ``Metabolinks`` before using ``pip``. These are listed below, but they
can be easily obtained by installing one of the "Scientific/Data Science Python" distributions.
One of these two products is highly recommended:

- `Anaconda Individual Edition <https://www.anaconda.com/products/individual>`_ (or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ followed by the necessary ``conda install``'s)
- `Enthought Deployment Manager <https://assets.enthought.com/downloads/edm/>`_ (followed by the creation of suitable Python environments)

The formal requirements are:

- Python 3.6 and above
- ``setuptools``, ``pip``, ``requests``, ``six``, ``pandas-flavor`` and ``pytest``

and, from the Python scientific ecossystem:

- ``numpy``, ``scipy``, ``matplotlib``, ``pandas`` and ``scikit-learn``

The installation of the ``Jupyter`` platform is also recommended since
the examples are provided as *Jupyter notebooks*.

