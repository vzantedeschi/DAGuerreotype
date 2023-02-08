# DAGuerreotype: DAG Learning on the Permutahedron


## Installation instructions

The script `linux-install.sh` installs everything, assuming to be in an environment with `python>=3.9` 
with dev packages installed. 

Preliminary commands to install from scratch on ubuntu, including creating a DGE environment:

```shell
sudo apt install python3.9
sudo apt install python3.9-venv
sudo apt-get install python3.9-dev
python3.9 -m venv ~/envs/DGE
source ~/envs/DGE/bin/activate
# install torch with gpu capability with cuda 11.6
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116  
chmod +x linux-install.sh
./linux-install.sh
```

`linux-install.sh` executes the following steps:

- Download and unpack eigen package: https://gitlab.com/libeigen/eigen
- Download, unpack and install lp-sparsemap package: https://github.com/deep-spin/lp-sparsemap
- Build the current project

For joint optimization add the joint argument

```bash
python -m daguerreo.run_model --joint
```

For bilevel leave default
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
Bilevel + LARS (important: must set --sparsifier to none)
```bash
python -m daguerreo.run_model --equations lars --sparsifier none --nogpu
```

### Implemented Modules

#### Edge Estimators
Implemented edge estimators are defined in `daguerreo/equations.py`:
1. `daguerreo.equations.LinearEquations`: differentiable linear layer X -> X W
2. `daguerreo.equations.NonlinearEquations`: differentiable one-hidden-layer network with leaky ReLU activation
3. `daguerreo.equations.LARSAlgorithm`: non-differentiable regressor as described in [Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game]{https://arxiv.org/abs/2102.13647}

New estimators should extend `daguerreo.equations.Equations`.