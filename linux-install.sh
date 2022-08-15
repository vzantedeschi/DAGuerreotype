cd ..
pip install --upgrade cython
wget --no-check-certificate --content-disposition https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
wget --no-check-certificate --content-disposition https://github.com/deep-spin/lp-sparsemap/archive/refs/heads/master.zip
unzip eigen-master.zip
unzip lp-sparsemap-master.zip
rm eigen-master.zip
rm lp-sparsemap-master.zip
export EIGEN_DIR=`readlink -f eigen-master`
cd lp-sparsemap-master
python setup.py build_clib
pip install -e .
cd ../dag-learning
python3 setup.py build_ext --inplace