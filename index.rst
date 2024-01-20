.. nanomcmc documentation master file, created by
   sphinx-quickstart on Wed Jan 10 10:24:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./_static/logo.svg
  :align: center
  :width: 100
  :alt: nanomcmc logo

``nanomcmc``
==============

``nanomcmc`` is a python package for extremely parallelized MCMC simulations built on pytorch. MCMC is a very versatile algorithm. It can be used to realize Boltzmann distributions, random walks, scramblers, dynamics, and many other applications. We have an initial state of the system, :math:`\mathbf{s}_0`. Then, MCMC is a two step process:

.. graphviz::
   :align: center

   digraph {
      rankdir=LR;
      node [shape=box];
      s0[label=<s<SUB>0</SUB>>];
      s1[label=<s<SUB>1</SUB>>];
      s2[label=<s<SUB>t</SUB>>];
      s0 -> s1;
      s1 -> s2 [style=dashed];
   }

Step 1: Propose a new state by sampling 

.. math::

    \mathbf{s}'_{t+1} \sim p(\mathbf{s}'_{t+1} \vert \mathbf{s}_{t})


The proposer can randomly flip bits, uniformly choose a new sample, etc.

Step 2: Accept or reject the new state using an acceptance rule

.. math::

    \mathbf{s}_{t+1} \sim a(\mathbf{s}_{t+1} \vert \mathbf{s}'_{t+1}, \mathbf{s}_{t})

.. graphviz::
   :align: center

   digraph {
      graph [ splines = false];
      rankdir=LR;
      node [shape=box];
      s0[label=<s<SUB>t</SUB>>];
      s1[label=<s'<SUB>1</SUB>>];
      s2[label=<s<SUB>t+1</SUB>>];
      s0 -> s1 [label="Propose"];
      s0 -> s1 [label=<r(s'<SUB>t+1</SUB> | s<SUB>t</SUB>)>];
      s1 -> s2 [label="Accept/Reject"];
      s1 -> s2 [label=<a(s<SUB>t+1</SUB> | s'<SUB>1</SUB>, s<SUB>t</SUB>)>];
   }

API
---
.. toctree::
   :maxdepth: 2

   modules

Usage
-----

Installation
~~~~~~~~~~~~

To use ``nanomcmc``, first install it using ``pip`` from the command line:

.. code-block:: console


   $ python -m venv .venv
   $ source .venv/bin/activate
   $ (.venv) python -m pip install git+https://github.com/nanometaml/mcmc.git

Or, clone the package and install it using ``pip`` from the command line:

.. code-block:: console

    $ git clone git+https://github.com/nanometaml/mcmc.git
    $ python -m pip install -e ./mcmc


Examples
--------
All of the following examples assume that you have imported ``nanomcmc``:

.. code-block:: python

    import nanomcmc as mcmc



Scrambler
~~~~~~~~~

Let's start with a simple example. We have a system with 3 binary variables. We want to jump around randomly to scramble the bits. 

.. code-block:: python

    # Initial state
    s_0 = torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.float32)

Uniform Scrambler 
^^^^^^^^^^^^^^^^^^^^^^

We can define a scrambler as follows. We want to randomly flip each bit with a probability of 0.5.


.. math::

    \mathbf{s}'_{t+1} \sim p(\mathbf{s}'_{t+1} \vert \mathbf{s}_{t}) = 2^-n 

Which is equivalent to choosing each bit with a fair coin,

.. math::

    \mathbf{s}'_{t+1, i} \sim \text{Bernoulli}(0.5)


.. code-block:: python

    # Random uniform proposer
    proposer = lambda s: torch.bernoulli(torch.ones_like(s) * 0.5)  

Acceptance Rule
^^^^^^^^^^^^^^^
To keep things simple, we'll always accept the new state.

Our acceptance rule is to always accept the new state,

.. math::

    a(\mathbf{s}_{t+1} \vert \mathbf{s}'_{t+1}, \mathbf{s}_{t}) = \delta(\mathbf{s}_{t+1} - \mathbf{s}'_{t+1})

.. math::

    \mathbf{s}_{t+1} = \mathbf{s}'_{t+1}

.. code-block:: python

    # Automatically accept all proposals
    acceptanceRule = lambda s, s_p: s_p  

MCMC
^^^^

We put it all together using the ``MCMC`` class:

.. code-block:: python

    scrambler = mcmc.MCMC(
        proposer=proposer, acceptanceRule=acceptanceRule, steps=1
    )

    scrambler(s_0)

    >>> tensor([[0., 1., 1.],
                [1., 1., 1.]])



Random Walk
~~~~~~~~~~~

Let's try a more interesting example. We have our same system with 3 binary variables and we want to perform a random walk.

.. code-block:: python

    # Automatically accept all proposals
    acceptanceRule = lambda s, s_p: s_p  

    def proposer(s):
        # Chooses a random bit flip
        s_f = torch.distributions.OneHotCategorical(probs=torch.ones_like(s) / s.shape[-1]).sample()
        # Flips the bit
        return torch.remainder(s + s_f, 2)

    # 1 step in the chain
    steps = 1

    # Initial state
    s_0 = torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.float32)

    scrambler = mcmc.MCMC(
        proposer=proposer, acceptanceRule=acceptanceRule, steps=steps
    )

    scrambler(s_0)

    >>> tensor([[1., 1., 1.],
                [0., 1., 1.]])

Notices how the output is only one step away from the input. Increase the number of steps to get a longer random walk and increase the Hamming distance.

Future Tutorials
----------------

1. Boltzmann Distributions
2. Quantum Annealing



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
