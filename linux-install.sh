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
export MACOS_DEPLOYMENT_TARGET=10.14  # on MacOS
export EIGEN_DIR=`readlink -f eigen-master`
cd lp-sparsemap-master
python setup.py build_clib # builds ad3 in-place
pip install -e . # builds lpsmap and creates a link
# other dependencies
cd ..
export PYTHONPATH="${PYTHONPATH}:`readlink -f lp-sparsemap-master`"
# wget --no-check-certificate --content-disposition https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# wget --no-check-certificate --content-disposition https://cran.r-project.org/src/contrib/Archive/SID/SID_1.0.tar.gz
cd DAGuerreotype
python setup.py build_ext --inplace
# install R and libraries to compute SID (optional)
sudo apt-get update &&
sudo apt-get install -y software-properties-common &&
sudo rm -rf /var/lib/apt/lists/*
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 &&
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9 &&
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
sudo apt-get update -y --fix-missing &&
sudo apt-get install -y --no-install-recommends r-base &&
sudo apt-get clean -y
Rscript -e 'install.packages("BiocManager")' &&
Rscript -e 'BiocManager::install(c("RBGL", "igraph", "graph"))' &&
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/fastICA/fastICA_1.2-2.tar.gz", repos=NULL, type="source")' &&
Rscript -e 'BiocManager::install(c("pcalg", "SID"))'
