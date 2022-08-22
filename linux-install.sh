pip install wheel
pip install -r requirements.txt
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
# other dependencies
cd ../
#wget --no-check-certificate --content-disposition https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
#wget --no-check-certificate --content-disposition https://cran.r-project.org/src/contrib/Archive/SID/SID_1.0.tar.gz
cd dag-learning
python setup.py build_ext --inplace