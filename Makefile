LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
SHELL = bash 
.SHELLFLAGS = -ec -o pipefail

# Install the dependencies
.PHONY: build
build: Pipfile.lock
	@echo Initializing environment...
	@pip install pipenv
	@mamba install s5cmd

# Download the datasets
.PHONY: all_datasets fgvcx_2018 fgvcx_2019 fgvcx_2021 build_old_tf get_models deploy
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
	@tar -xvf ./training/base_models/checkpoints/swin_base_224/swin_base_224.tgz -C ./training/base_models/checkpoints/swin_base_224/
	@tar -xvf ./training/base_models/checkpoints/swin_base_384/swin_base_384.tgz -C ./training/base_models/checkpoints/swin_base_384/
	@tar -xvf ./training/base_models/checkpoints/swin_large_224/swin_large_224.tgz -C ./training/base_models/checkpoints/swin_large_224/
	@tar -xvf ./training/base_models/checkpoints/swin_large_384/swin_large_384.tgz -C ./training/base_models/checkpoints/swin_large_384/
	@tar -xvf ./training/base_models/checkpoints/swin_small_224/swin_small_224.tgz -C ./training/base_models/checkpoints/swin_small_224/
	@tar -xvf ./training/base_models/checkpoints/swin_tiny_224/swin_tiny_224.tgz -C ./training/base_models/checkpoints/swin_tiny_224/
	@echo "Finished downloading model weights..."

build_old_tf: ./training/base_models/checkpoints/Pipfile.lock
	@echo "Building old tensorflow environment..."
	@cd ./training/base_models/checkpoints && pipenv install --skip-lock
	@echo "Finished building old tensorflow environment..."

resave_base_model_weights: build_old_tf download_model_weights
	@echo "Resaving model weights..."
	@cd ./training/base_models/checkpoints && pipenv run python ../../../scripts/resave_base_model_weights.py
	@rm -rf ./training/base_models/checkpoints
	@echo "Finished resaving model weights..."

# get_models:
# 	@echo "Downloading models..."
# 	@mkdir -p ./training/base_models
# 	@gsutil -m cp -r gs://mush-img-repo/base_models/* ./training/base_models/
# 	@echo "Finished downloading models..."

get_deploy_model: ./deploy/model/
	@echo "Downloading model..."
	@wandb artifact get model-registry/Mushroom-Classifier:latest --root ./deploy/model

deploy: get_deploy_model
	@echo "Deploying model..."
	@cd deploy && pipenv run python app.py & ngrok http 5000
	@echo "Finished deploying model..."