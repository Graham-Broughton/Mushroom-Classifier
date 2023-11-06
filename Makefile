LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
SHELL = bash 
.SHELLFLAGS = -ec -o pipefail
MODEL_DIR=./deploy/model
MODEL_MARKER=$(MODEL_DIR)/.downloaded

include .env
export

# Install the dependencies
.PHONY: all build all_datasets fgvcx_2018 fgvcx_2019 fgvcx_2021 build_old_tf resave_base_model_weights get_base_models get_deploy_model deploy help

all: help

build: Pipfile.lock
	@echo Initializing environment...
	@pip install pipenv
	@mamba install s5cmd

dotenv:
	./scripts/dotenvs.sh

#################################################
### Data
#################################################

# Download the datasets - all or individual years
all_datasets: fgvcx_2018 fgvcx_2019 fgvcx_2021
	@echo "Finished downloading and extracting datasets..."

the_two_good_ones: fgvcx_2018 fgvcx_2021

fgvcx_2018: build scripts/get_data.sh
	@echo "Downloading datasets fgvcx 2018..."
	@bash scripts/get_data.sh -y 2018

	@echo "Extracting datasets & removing tar.gz files..."
	@for f in data/raw/2018/*.tar.gz; do tar -xvf $${f} -C data/raw/2018/; done;
	@rm -rf data/raw/2018/*.tar.gz

	@echo "Finished extracting datasets..."

fgvcx_2019: build scripts/get_data.sh
	@echo "Downloading datasets fgvcx 2019..."
	@bash scripts/get_data.sh -y 2019

	@echo "Extracting datasets & removing tar.gz files..."
	@for f in data/raw/2019/*.tar.gz; do tar -xzf $${f} -C data/raw/2019/; done
	@rm -rf data/raw/2019/*.tar.gz

	@echo "Finished extracting datasets..."

fgvcx_2021: build scripts/get_data.sh
	@echo "Downloading datasets fgvcx 2021..."
	@bash scripts/get_data.sh -y 2021

	@echo "Extracting datasets & removing tar.gz files..."
	@for f in data/raw/2021/*.tar.gz; do tar -xzf $${f} -C data/raw/2021/; done
	@rm -rf data/raw/2021/*.tar.gz

	@echo "Finished extracting datasets..."

#################################################
### TFRecords
#################################################

# Convert the images to tfrecords with many user options
# IMG_DIR ?= $(shell bash -c 'read -p "Where are the images located (OPTIONAL, default: ./data/)? " image_path; echo $$image_path')
# TFREC_DIR ?= $(shell bash -c 'read -p "Where should the tfrecords be saved (OPTIONAL, default: ./data/)? " tfrecord_path; echo $$tfrecord_path')
# TRAIN_RECS ?= $(shell bash -c 'read -p "Number of train image tfrecords (OPTIONAL, default: 107)? " num_train_records; echo $$num_train_records')
# VAL_RECS ?= $(shell bash -c 'read -p "Number of validation image tfrecords (OPTIONAL, default: 5)? " num_val_records; echo $$num_val_records')
# IMG_SIZES ?= $(shell bash -c 'read -p "Image height and width (REQUIRED, default: 224, 224)? " image_size; echo $$image_size')
# MULTIPROCESSING ?= $(shell bash -c 'read -p "Use multiprocessing (OPTIONAL, default: True)? " multiprocessing; echo $$multiprocessing')
tfrecords: build scripts/create_tfrecords.sh
	@echo "Creating tfrecords..."
	@: $(eval IMG_DIR := $(shell bash -c 'read -p "Where are the images located (OPTIONAL, default: ./data/)? " image_path; echo $$image_path'))
	@: $(eval TFREC_DIR := $(shell bash -c 'read -p "Where should the tfrecords be saved (OPTIONAL, default: ./data/)? " tfrecord_path; echo $$tfrecord_path'))
	@: $(eval TRAIN_RECS := $(shell bash -c 'read -p "Number of train image tfrecords (OPTIONAL, default: 107)? " num_train_records; echo $$num_train_records'))
	@: $(eval VAL_RECS := $(shell bash -c 'read -p "Number of validation image tfrecords (OPTIONAL, default: 5)? " num_val_records; echo $$num_val_records'))
	@: $(eval IMG_SIZES := $(shell bash -c 'read -p "Image height and width (REQUIRED, default: 224, 224)? " image_size; echo $$image_size'))
	@: $(eval MULTIPROCESSING := $(shell bash -c 'read -p "Use multiprocessing (OPTIONAL, default: True)? " multiprocessing; echo $$multiprocessing'))
	@bash scripts/create_tfrecords.sh -d $(IMG_DIR) -p $(TFREC_DIR) -t $(TRAIN_RECS) -v $(VAL_RECS) -s $(IMG_SIZES) -m $(MULTIPROCESSING)
	@echo "Finished creating tfrecords..."

#################################################
### Models
#################################################

# Download the model weights for Tensorflow from a GitHub repo
download_model_weights:
	@echo "Downloading model weights..."
	@mkdir -p ./training/base_models/checkpoints/swin_base_224/
	@mkdir -p ./training/base_models/checkpoints/swin_base_384/
	@mkdir -p ./training/base_models/checkpoints/swin_large_224/
	@mkdir -p ./training/base_models/checkpoints/swin_large_384/
	@mkdir -p ./training/base_models/checkpoints/swin_small_224/
	@mkdir -p ./training/base_models/checkpoints/swin_tiny_224/

	@wget -O ./training/base_models/checkpoints/swin_base_224/swin_base_224.tgz https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_base_224.tgz
	@wget -O ./training/base_models/checkpoints/swin_base_384/swin_base_384.tgz https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_base_384.tgz
	@wget -O ./training/base_models/checkpoints/swin_large_224/swin_large_224.tgz https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_large_224.tgz
	@wget -O ./training/base_models/checkpoints/swin_large_384/swin_large_384.tgz https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_large_384.tgz
	@wget -O ./training/base_models/checkpoints/swin_small_224/swin_small_224.tgz https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_small_224.tgz
	@wget -O ./training/base_models/checkpoints/swin_tiny_224/swin_tiny_224.tgz https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_tiny_224.tgz

	@echo "extracting model weights..."
	@tar -xvf ./training/base_models/checkpoints/swin_base_224/swin_base_224.tgz -C ./training/base_models/checkpoints
	@tar -xvf ./training/base_models/checkpoints/swin_base_384/swin_base_384.tgz -C ./training/base_models/checkpoints
	@tar -xvf ./training/base_models/checkpoints/swin_large_224/swin_large_224.tgz -C ./training/base_models/checkpoints
	@tar -xvf ./training/base_models/checkpoints/swin_large_384/swin_large_384.tgz -C ./training/base_models/checkpoints
	@tar -xvf ./training/base_models/checkpoints/swin_small_224/swin_small_224.tgz -C ./training/base_models/checkpoints
	@tar -xvf ./training/base_models/checkpoints/swin_tiny_224/swin_tiny_224.tgz -C ./training/base_models/checkpoints

	@rm -rf ./training/base_models/checkpoints/*.tgz
	@echo "Finished downloading model weights..."

# Build the tensorflow 2.10.0 environment, it is needed to resave the model weights into a SavedModel format
build_old_tf: ./training/base_models/checkpoints/Pipfile.lock
	@echo "Building tensorflow 2.10.0 environment..."
	@cd ./training/base_models/checkpoints && pipenv install --skip-lock
	@echo "Finished building tensorflow 2.10.0 environment..."

resave_base_model_weights:  download_model_weights build_old_tf
	@echo "Resaving model weights..."
	@cd ./training/base_models/checkpoints && pipenv run resave
	@rm -rf ./training/base_models/checkpoints
	@echo "Finished resaving model weights..."

# Download the re-saved models
get_base_models:
	@echo "Downloading models..."
	@mkdir -p ./training/base_models
	@gsutil -m cp -r "gs://$(GCS_REPO)/$(GCS_BASE_MODELS)/*"" ./training/base_models/
	@echo "Finished downloading models..."

#################################################
### Deploy
#################################################

# The target that depends on MODEL_MARKER
get_deploy_model: $(MODEL_MARKER)
	@echo "Model is already downloaded."

# Deploy the model to a local server using ngrok
deploy: get_deploy_model
	@echo "Deploying model..."
	@cd deploy && pipenv run python app.py & 
	@ngrok http 5000
	@echo "Finished deploying model..."

# Download the latest registered model from wandb if it doesn't exist
$(MODEL_MARKER):
	@echo "Downloading model..."
	@mkdir -p $(MODEL_DIR)
	@pipenv run wandb artifact get model-registry/$(WANDB_REGISTERED_MODEL):latest --root $(MODEL_DIR)
	@touch $(MODEL_MARKER)

#################################################
### Terraform
#################################################

terraform:
	cp .env infrastructure/.env
	cd infrastructure && terraform init
	cd infrastructure && ./terraform-apply.sh

#################################################
### Help
#################################################

help:
	@echo "========================================="
	@echo "build:             					- Install the dependencies"
	@echo "all_datasets:      					- Download all dataset years"
	@echo "fgvcx_2018:        					- Download the 2018 dataset"
	@echo "fgvcx_2019:        					- Download the 2019 dataset"
	@echo "fgvcx_2021:        					- Download the 2021 dataset"
	@echo "download_model_weights: 				- Download Tensorflow model weights from a GitHub repo"
	@echo "build_old_tf:      					- Build the tensorflow 2.10.0 environment, needed to resave model weights into SavedModel format"
	@echo "resave_base_model_weights:				- Resave the model weights into a SavedModel format"
	@echo "get_base_models:   					- Download the re-saved models"
	@echo "get_deploy_model:  					- Download the latest registered model from wandb"
	@echo "deploy:            					- Deploy the model to a local server using ngrok"
	@echo "tfrecords:         					- Convert the images to tfrecords with many user options"
	@echo "========================================="

#################################################
