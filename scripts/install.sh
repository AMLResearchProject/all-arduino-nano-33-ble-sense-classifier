#!/bin/bash

FMSG="Acute Lymphoblastic Leukemia Arduino Nano 33 BLE Sense Classifier trainer installation terminated!"

echo "This script will install Acute Lymphoblastic Leukemia Arduino Nano 33 BLE Sense Classifier."
echo "HINT: This script assumes Ubuntu 20.04."
echo "WARNING: This script assumes you have not already installed the oneAPI Basekit."
echo "WARNING: This script assumes you have not already installed the oneAPI AI Analytics Toolkit."
echo "WARNING: This script assumes you have an Intel GPU."
echo "WARNING: This script assumes you have already installed the Intel GPU drivers."
echo "HINT: If any of the above are not relevant to you, please comment out the relevant sections below before running this installation script."

read -p "Proceed (y/n)? " proceed
if [ "$proceed" = "Y" -o "$proceed" = "y" ]; then
	# Comment out the following if you have already installed oneAPI Basekit
	wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O - | sudo apt-key add -
	echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
	sudo apt update
	sudo apt install intel-basekit
	sudo apt -y install cmake pkg-config build-essential
	echo 'source /opt/intel/oneapi/setvars.sh' >> ~/.bashrc
	source ~/.bashrc
	# Comment out the following if you have already installed oneAPI AI Analytics
	sudo apt install intel-aikit
	# Comment out the following if you have already installed the Intel GPU drivers
	# or do not have an Intel GPU on your training device
	sudo apt-get install -y gpg-agent wget
	wget -qO - https://repositories.intel.com/graphics/intel-graphics.key |
	sudo apt-key add -
	sudo apt-add-repository \
	'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
	sudo apt-get update
	sudo apt-get install \
	intel-opencl-icd \
	intel-level-zero-gpu level-zero \
	intel-media-va-driver-non-free libmfx1
	stat -c "%G" /dev/dri/render*
	groups ${USER}
	sudo gpasswd -a ${USER} render
	newgrp render
	sudo usermod -a -G video ${USER}
	# The following wil install all other required packages
	conda create -n all-nano-33-ble-sense -c intel intel-aikit-tensorflow
	conda activate all-nano-33-ble-sense
	conda install jupyter
	conda install nb_conda
	conda install -c conda-forge mlxtend
	conda install matplotlib
	conda install Pillow
	conda install opencv
	conda install scipy
	conda install scikit-learn
	conda install scikit-image
else
	echo $FMSG;
	exit 1
fi