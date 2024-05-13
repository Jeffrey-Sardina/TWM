#!/bin/bash

# Update package list
apt-get update

# Install conda
apt-get install -y wget
if [[ ! -d "/root/miniconda3" ]]
then
    miniconda_name="Miniconda3-latest-Linux-x86_64.sh"
    wget https://repo.anaconda.com/miniconda/$miniconda_name
    chmod u+x $miniconda_name
    ./$miniconda_name -b
    rm $miniconda_name
fi

# Install Python packages
export PATH=$PATH:~/miniconda3/bin
apt install -y git # needed for PyKEEN internals
apt install -y sqlite # needed for optuna, used by PyKEEN
conda create -n "twig" python=3.9 pip
conda run --no-capture-output -n twig pip install torch torchvision torchaudio torcheval
conda run --no-capture-output -n twig pip install pykeen numba discord flask
conda init bash

# Install Node.js and NPM for Gephi Lite
apt update
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install v20.12.0
apt install -y npm
npm install -g npm@latest
npm install typescript@5.3
cd twig/TWM/
git clone https://github.com/gephi/gephi-lite.git
cd gephi-lite/
npm install
cd ../../..

# Add to conda to .bashrc
echo "conda activate twig" >> ~/.bashrc
