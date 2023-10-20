SHELL = bash 
.SHELLFLAGS = -ec -o pipefail

# Install the dependencies
.PHONY: build
build: Pipfile.lock
	@echo Initializing environment...
	@pip install pipenv
	@mamba install s5cmd

# Download the datasets
.PHONY: all_datasets fgvcx_2018 fgvcx_2019 fgvcx_2021
all_datasets: fgvcx_2018 fgvcx_2019 fgvcx_2021
	@echo "Finished downloading and extracting datasets..."

fgvcx_2018: build scripts/get_data.sh
	@echo "Downloading datasets fgvcx 2018..."
	@bash scripts/get_data.sh -y 2018
	@echo "Extracting datasets..."
	@for f in data/raw/2018/*.tar.gz; do tar -xvf $${f} -C data/raw/2018/; done;
	@echo "Removing tar.gz files..."
	@rm -rf data/raw/2018/*.tar.gz
	@echo "Finished extracting datasets..."

fgvcx_2019: build scripts/get_data.sh
	@echo "Downloading datasets fgvcx 2019..."
	@bash scripts/get_data.sh -y 2019
	@echo "Extracting datasets..."
	@for f in data/raw/2019/*.tar.gz; do tar -xzf $${f} -C data/raw/2019/; done
	@rm -rf data/raw/2019/*.tar.gz
	@echo "Finished extracting datasets..."

fgvcx_2021: build scripts/get_data.sh
	@echo "Downloading datasets fgvcx 2021..."
	@bash scripts/get_data.sh -y 2021
	@echo "Extracting datasets..."
	@for f in data/raw/2021/*.tar.gz; do tar -xzf $${f} -C data/raw/2021/; done
	@rm -rf data/raw/2021/*.tar.gz
	@echo "Finished extracting datasets..."

IMG_DIR ?= $(shell bash -c 'read -p "Where are the images located (OPTIONAL, default: ./data/)? " image_path; echo $$image_path')
TFREC_DIR ?= $(shell bash -c 'read -p "Where should the tfrecords be saved (OPTIONAL, default: ./data/)? " tfrecord_path; echo $$tfrecord_path')
TRAIN_RECS ?= $(shell bash -c 'read -p "Number of train image tfrecords (OPTIONAL, default: 107)? " num_train_records; echo $$num_train_records')
VAL_RECS ?= $(shell bash -c 'read -p "Number of validation image tfrecords (OPTIONAL, default: 5)? " num_val_records; echo $$num_val_records')
IMG_SIZES ?= $(shell bash -c 'read -p "Image height and width (REQUIRED, default: 224, 224)? " image_size; echo $$image_size')
MULTIPROCESSING ?= $(shell bash -c 'read -p "Use multiprocessing (OPTIONAL, default: True)? " multiprocessing; echo $$multiprocessing')
tfrecords: build scripts/create_tfrecords.sh
	@echo "Creating tfrecords..."
	@bash scripts/create_tfrecords.sh -d $(IMG_DIR) -p $(TFREC_DIR) -t $(TRAIN_RECS) -v $(VAL_RECS) -s $(IMG_SIZES) -m $(MULTIPROCESSING)
	@echo "Finished creating tfrecords..."
