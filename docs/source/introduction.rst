Introduction
============

``oasi-ozone-forecaster`` is a package for the forecast of the next day ozone maximum hourly concentration in Ticino, Switzerland. These forecasts are calculated twice: the night before for the upcoming day and the morning of the same day for the present day.
The forecasts are achieved through a ML approach which is trained on a set of both measured and forecasted meteorological values from 2015 to 2021. Please consults the `openly accessible published paper <https://www.sciencedirect.com/science/article/pii/S0169207021001199/>`_ on the subject for more information.

Basic instructions
******************

To properly configure the installation, refer to the following basic instructions:

1. Clone the repo: ``git clone https://github.com/supsi-dacd-isaac/oasi-ozone-forecaster.git``
2. Move to the newly created directory: ``cd oasi-ozone-forecaster``
3. Create ``logs`` directory: ``mkdir logs/``
4. Create a virtual environment: ``python3 -m venv venv/``
5. Install the necessary packages: 

.. code-block:: python

   venv/bin/python3 -m pip install numpy
   venv/bin/python3 -m pip install -r requirements.txt

Necessary  adjustaments
***********************

The skgarden package is severely outdated, thus some manual modifications need to be applied in folder ``/venv/lib64/python3.8/site-packages/skgarden/``:

1. In file ``quantile/ensemble.py``, line 40: add ``sample_weight`` to method arguments
#. In file ``quantile/ensemble.py``, line 79: add ``sample_weight`` to method call
#. In file ``quantile/tree.py``, lines 221 and 232: remove ``presort=False`` option to clean stdout output (optional)
#. In file ``mondrian/ensemble/forest.py``, line 96 and 229: add ``sample_weight`` to method arguments
#. In file ``mondrian/ensemble/forest.py``, line 121 and 254: add ``sample_weight`` to method call

You are now ready to execute the scripts. Proceed to :ref:`Main scripts` and the descriptions therein.
