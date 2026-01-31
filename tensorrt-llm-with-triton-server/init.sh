#! /bin/bash

sudo apt update
sudo apt install python3.10-venv
cd ~
python3 -m venv venv
source venv/bin/activate
pip install "huggingface_hub[cli]"
sudo apt-get install git-lfs
git clone https://github.com/NVIDIA/TensorRT-LLM.git
