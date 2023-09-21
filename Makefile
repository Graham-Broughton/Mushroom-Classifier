SHELL = bash 
.SHELLFLAGS = -ec -o pipefail

# Install the dependencies
.PHONY: build
build: Pipfile.lock
	echo Initializing environment...
	@pip install pipenv
	@pipenv install --ignore-pipfile
	@pipenv run mamba install s5cmd

# Download the datasets
.PHONY: all_datasets
all_datasets: fgvcx_2018 fgvcx_2019 fgvcx_2021
	echo "Finished downloading and extracting datasets..."

fgvcx_2018: build scripts/get_data.sh
	echo "Downloading datasets fgvcx 2018..."
	@pipenv run bash scripts/get_data.sh -y 2018
	echo "Extracting datasets..."
	@for f in data/raw/2018/*.tar.gz; do tar -xvf $${f} -C data/raw/2018/; done;
	@rm -rf data/raw/2018/*.tar.gz
	echo "Finished extracting datasets..."

fgvcx_2019: build scripts/get_data.sh
	echo "Downloading datasets fgvcx 2019..."
	@pipenv run bash scripts/get_data.sh -y 2019
	echo "Extracting datasets..."
	@for f in data/raw/2019/*.tar.gz; do tar -xzf $${f} -C data/raw/2019/; done
	@rm -rf data/raw/2019/*.tar.gz
	echo "Finished extracting datasets..."

fgvcx_2021: build scripts/get_data.sh
	echo "Downloading datasets fgvcx 2021..."
	@pipenv run bash scripts/get_data.sh -y 2021
	echo "Extracting datasets..."
	@for f in data/raw/2021/*.tar.gz; do tar -xzf $${f} -C data/raw/2021/; done
	@rm -rf data/raw/2021/*.tar.gz
	echo "Finished extracting datasets..."
