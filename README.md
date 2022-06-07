# discrete DAG learning

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