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

fgvcx_2018: build get_data.sh
	echo "Downloading datasets fgvcx 2018..."
	@pipenv run bash get_data.sh -y 2018
	echo "Extracting datasets..."
	cd 2018
	@for f in ./*.tar.gz; do tar -xzf --remove-files $$f -C ./; done
	echo "Finished extracting datasets..."
	cd ../

fgvcx_2019: build get_data.sh
	echo "Downloading datasets fgvcx 2019..."
	@pipenv run bash get_data.sh -y 2019
	echo "Extracting datasets..."
	cd 2019
	@for f in ./*.tar.gz; do tar -xzf --remove-files $$f -C ./; done
	echo "Finished extracting datasets..."
	cd ../

fgvcx_2021: build get_data.sh
	echo "Downloading datasets fgvcx 2021..."
	@pipenv run bash get_data.sh -y 2021
	echo "Extracting datasets..."
	cd 2021
	@for f in ./*.tar.gz; do tar --remove-files -xzf $$f -C ./; done
	echo "Finished extracting datasets..."
	cd ../
