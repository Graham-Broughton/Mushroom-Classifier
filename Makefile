SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

# Install the dependencies
.PHONY: build
build: Pipfile.lock
	echo Initializing environment...
	@pip install pipenv
	@pipenv install --ignore-pipfile
	@pipenv shell
	@curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
	@./Mambaforge-$(uname)-$(uname -m).sh
	@mamba install s5cmd

# Download the datasets
.PHONY: all_datasets
all_datasets: build get_data.sh
	echo "Downloading datasets..."
	@./get_data.sh -a

fgvcx_2018: build get_data.sh
	echo "Downloading datasets fgvcx 2018..."
	@./get_data.sh -y 2018

fgvcx_2019: build get_data.sh
	echo "Downloading datasets fgvcx 2019..."
	@./get_data.sh -y 2019

fgvcx_2021: build get_data.sh
	echo "Downloading datasets fgvcx 2021..."
	@./get_data.sh -y 2021

fgvcx_extras: build get_data.sh
	echo "Downloading datasets fgvcx extras..."
	@./get_data.sh -y extras