# discrete DAG learning

NOTE: big refactoring on this branch, it's still WIP.

Divided the old modules.py into three modules. Removed bernoulli.py (now in sparsifiers.py)

For joint optimization add the joint argument

```bash
python -m daguerreo.run_model --joint
```

For bileve leave default
```bash
python -m daguerreo.run_model
```

This will run on the Sachs dataset.
For small synthetic ER with Gaussian noise use the following
```bash
python -m daguerreo.run_model --joint --dataset synthetic
python -m daguerreo.run_model --dataset synthetic
```

Run with Sparsemax 
```bash
python -m daguerreo.run_model --structure tk_sp_max --joint --dataset synthetic
python -m daguerreo.run_model --structure tk_sp_max --dataset synthetic
```

Bilevel + LARS is working, evaluation pipeline to be finalized

## TODOs coding
- ~~Top-K SparseMax~~
- HPO pipeline
- storing results outside wandb
- metrics with Markov eqiv. classes
- write a script that installs everything needed!!!!
- check the bias issue! (no bias in linear & non-linear models)!!
- etc....

## TODOs experiments

- sparsemax 
- bilvel vs joint (fair comparrison with same:(i) runtime! (ii) number of "seen" DAGS)

## Installation instructions

Get `python>=3.9`, on Mac `python=3.8` did not work.

- Download eigen package: https://gitlab.com/libeigen/eigen
- Unpack eigen and take note of the main folder (e.g. `/Users/[....]/eigen-3.4.0`)
- Download lp-sparsemap package: https://github.com/deep-spin/lp-sparsemap
- Run following commands in the lp-sparsemap main folder
```bash
pip install --upgrade cython
export MACOS_DEPLOYMENT_TARGET=10.14  # on MacOS
export EIGEN_DIR=/path/to/eigen
python setup.py build_clib  # builds ad3 in-place
pip install -e .            # builds lpsmap and creates a link
```
- return to this project main folder and run
```bash
python3 setup.py build_ext --inplace
```

``

