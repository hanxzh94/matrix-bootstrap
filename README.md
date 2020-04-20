# Bootstrapping Matrix Quantum Mechanics

This is the code for bootstrap numerics related to arXiv: ... 

Here `algebra.py` contains some sparse matrix linear algebra routines for solving the linear equations, `optimize.py` uses a sequential semidefinite programming algorithm to do gradient descent with constraints, `demo.py` has some simple demonstrations of the numerics, `solver.py` is the main code for generating relations between observables and the bootstrap matrix, `trace.py` includes a helper class for dealing with trace operators and `utility.py` has miscellaneous functions for debugging.

To use the code, please follow these steps.

1. Install Anaconda (https://docs.anaconda.com/anaconda/install/);

2. Create a new environment (named bootstrap here) with the following command:

   `conda create -n bootstrap python=3.7 numpy=1.17 scipy=1.4.1`
   
   `conda activate bootstrap`
   
3. Install `cvxpy` (https://www.cvxpy.org/install/index.html). And remember to install the SCS solver:

   `conda install -c cvxgrp --yes ecos scs multiprocess`

4. Install `sparseqr` (https://github.com/yig/PySPQR). Before installing `sparseqr` you will need to have `suitesparse` (http://faculty.cse.tamu.edu/davis/suitesparse.html):

   `conda install -c conda-forge suitesparse`

5. Run `python demo.py 3 1.0` as a test. The demo should run the two-matrix bootstrap with L = 3 and g = 1.0. The result energy should be close to 2.32.

6. Feel free to change the variable `case` in `demo.py` for other demonstrations! 
