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

# TODO 

### Theory

- formalize the "continuity" argument of the MAP @luca
- rewrite the "structure" subsection of the paper following Vlad's notes  @vlad
- etc.

### Code
- ~~Top-K SparseMax~~
- remove temperature in spmax and max  @luca
- HPO pipeline @vale
  - tune on synthetic datasets (maybe from a distribution over synth datasets)
- storing results outside wandb  @luca
- metrics with Markov eqiv. classes !!!!!! @matt
- debiased LARS @vale
- SID without R 
- write a script that installs everything needed!!!!
- check the bias issue! (no bias in linear & non-linear models) [low priority]
- etc....

### Experiments

- synthetic
  - compare distributions of sparsemap/max over graphs vs the "true" distribution that is a sparse distribution 
     with mass only on graphs in the same Markov class as the ground truth graph
- real world data
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

