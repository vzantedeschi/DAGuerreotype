# discrete DAG learning

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

---
old stuff
---

## dependencies
1. Eigen

Download and install without cmake [Eigen](https://gitlab.com/libeigen/eigen/)

2. Cython
```bash 
pip install --upgrade cython
```

3. Lp-Sparsemap

In-place install from source [lpsmap](https://github.com/deep-spin/lp-sparsemap)

### DAG generation
For data generation, we additionally need `scipy, python-igraph` and `sklearn`. 

wandb


## setup

```bash
python3 setup.py build_ext --inplace
```


