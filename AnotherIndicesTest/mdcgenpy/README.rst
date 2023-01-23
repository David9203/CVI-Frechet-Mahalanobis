|docs|

mdcgenpy
========

mdcgenpy is a **M**\ ultidimensional **D**\ ataset for **C**\ lustering **Gen**\ erator.
This tool is aimed at researchers looking for synthetic datasets, in particular for testing clustering algorithms.
A variety of customization options are available, in order to allow for a wide range of use cases.

Using the generator is simple, and can even be used without parameters:

.. code-block:: python

    import mdcgenpy

    # Initialize cluster generator (all parameters are optional)
    cluster_gen = mdcgenpy.clusters.ClusterGenerator()

    # Get tuple with a numpy array with samples and another with labels
    data = cluster_gen.generate_data()

Installation
------------

The package can be installed with pip:

.. code-block:: bash

    $ pip install git+https://github.com/CN-TU/mdcgenpy

Generating data outside Python
------------------------------

It is also possible to use mdcgenpy without knowing python.

To do this, you just need to give as input a JSON file (check specification details `here <https://mdcgenpy.readthedocs.io/en/latest/json_format.html>`_).
Using the ``mdcgenpy.py`` script, the output will be sent in CSV format to stdout.

Example:

.. code-block:: bash

    $ ./mdcgenpy.py input_parameters.json > output.csv


Features
--------

- Efficient code, compatible with Python 2 and Python 3.
- Various possible distributions for the clusters are available out-of-the-box, and custom distributions are also
  allowed.
- Parameters allow for control over the overlap of the clusters, outliers, noise, correlation inside each cluster, etc.

Documentation
-------------

Documentation for this project can be found at `https://mdcgenpy.readthedocs.io`.

.. |docs| image:: https://readthedocs.org/projects/mdcgenpy/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://mdcgenpy.readthedocs.io/en/latest/?badge=latest

