TARGET_DATA_DIR="./training/data/raw"
S3_BASE="s3://ml-inat-competition-datasets"
SHELL = bash 
.SHELLFLAGS = -ec -o pipefail
MODEL_DIR=./mush_app/model
MODEL_MARKER=$(MODEL_DIR)/.downloaded
PROJECT_ID := $(shell gcloud config get-value project)
HOSTNAME := us-central1-docker.pkg.dev
ARTIFACT_REPO := mushroom-classifier-deploy
IMAGE_NAME := model-image
GCR_TAG := ${HOSTNAME}/${PROJECT_ID}/${ARTIFACT_REPO}/${IMAGE_NAME}

include .env
export

# Install the dependencies
.PHONY: all build init all_datasets datasets_of_interest fgvcx_2018 fgvcx_2019 fgvcx_2021 preprocess_data tfrecords download_model_weights get_deploy_model deploy help

all: help

build: pyproject.toml
	@echo Initializing environment...
	@conda create -n py311 python=3.11 -y
	@poetry install --no-root
	@mamba install s5cmd -y

dotenv:
	./scripts/dotenvs.sh

init: build dotenv
	@mkdir logs
	@echo "Finished initializing environment..."

#################################################
### Data
#################################################

# Download the datasets - all or individual years
all_datasets: fgvcx_2018 fgvcx_2019 fgvcx_2021
	@echo "Finished downloading and extracting datasets..."

datasets_of_interest: fgvcx_2018 fgvcx_2021
	@echo "Finished downloading and extracting datasets..."
	@echo "Removing uneeded images..."

fgvcx_2018:
	@echo "Downloading datasets fgvcx 2018..."
	@bash scripts/get_data.sh -y 2018
	@echo "Extracting datasets & removing tar.gz and zip files for 2018 data..."
	@for f in training/data/raw/2018/*.tar.gz; do tar -xzf $${f} -C training/data/raw/2018/ && rm $${f}; done
	@unzip -q training/data/raw/2018/*.zip -d training/data/raw/2018/ && rm training/data/raw/2018/*.zip
	@echo "Finished extracting 2018 dataset..."
	
fgvcx_2019: 
	@echo "Downloading datasets fgvcx 2019..."
	@bash scripts/get_data.sh -y 2019
	@for f in training/data/raw/2019/*.tar.gz; do tar -xzf $${f} -C training/data/raw/2019/ && rm $${f}; done
	@echo "Finished extracting 2019 dataset..."

fgvcx_2021: 
	@echo "Downloading datasets fgvcx 2021..."
	@bash scripts/get_data.sh -y 2021
	@echo "Extracting datasets & removing tar.gz and zip files for 2021 data..."
	@rm training/data/raw/2021/train_mini*
	@for f in training/data/raw/2021/*.tar.gz; do tar -xzf $${f} -C training/data/raw/2021/ && rm $${f}; done
	@echo "Finished extracting 2021 dataset..."

#################################################
### TFRecords
#################################################

preprocess_data:
	@echo "Removing unused images..."
	@poetry run python scripts/delete_unused_images.py $(TARGET_DATA_DIR)
	@echo "Preprocessing data..."
	@poetry run python training/src/data_processing/train_preprocessing.py
	@echo "Finished preprocessing data..."

tfrecords: preprocess_data
	@echo "Creating tfrecords..."
	@: $(eval IMG_DIR := $(shell bash -c 'read -p "Where are the images located (OPTIONAL, default: ./training/data/raw)? " image_path; echo $$image_path'))
	@: $(eval TFREC_DIR := $(shell bash -c 'read -p "Where should the tfrecords be saved (OPTIONAL, default: ./training/data/)? " tfrecord_path; echo $$tfrecord_path'))
	@: $(eval TRAIN_RECS := $(shell bash -c 'read -p "Number of train image tfrecords (OPTIONAL, default: 100)? " num_train_records; echo $$num_train_records'))
	@: $(eval VAL_RECS := $(shell bash -c 'read -p "Number of validation image tfrecords (OPTIONAL, default: 5)? " num_val_records; echo $$num_val_records'))
	@: $(eval IMG_SIZES := $(shell bash -c 'read -p "Image height and width (REQUIRED, default: 256)? " image_size; echo $$image_size'))
	@: $(eval MULTIPROCESSING := $(shell bash -c 'read -p "Use multiprocessing (OPTIONAL, default: True)? " multiprocessing; echo $$multiprocessing'))
	@python training/src/data_processing/tfrecords.py -d $(IMG_DIR) -p $(TFREC_DIR) -t $(TRAIN_RECS) -v $(VAL_RECS) -s $(IMG_SIZES) -m $(MULTIPROCESSING)
	@echo "Finished creating tfrecords..."

#################################################
### Models
#################################################

# Download the model weights for Tensorflow from a GitHub repo
download_model_weights:
	@echo "Downloading model weights..."
	@mkdir -p ./training/base_models/swin_base_224
	@mkdir -p ./training/base_models/swin_base_384
	@mkdir -p ./training/base_models/swin_large_224
	@mkdir -p ./training/base_models/swin_large_384
	@mkdir -p ./training/base_models/swin_small_224
	@mkdir -p ./training/base_models/swin_tiny_224

	@gsutil -m cp -r "gs://$(GCS_REPO)/$(GCS_BASE_MODELS)/*" ./training/base_models
	@echo "Finished downloading model weights..."

#################################################
### Deploy
#################################################

# The target that depends on MODEL_MARKER
get_deploy_model: $(MODEL_MARKER)
	@echo "Model is already downloaded."

# Deploy the model to a local server using ngrok
deploy: get_deploy_model
	@echo "Deploying model..."
	@cd mush_app && pipenv run python app.py & 
	@ngrok http 5000
	@echo "Finished deploying model..."

# Download the latest registered model from wandb if it doesn't exist
$(MODEL_MARKER):
	@echo "Downloading model..."
	@mkdir -p $(MODEL_DIR)
	@pipenv run wandb artifact get model-registry/$(WANDB_REGISTERED_MODEL):latest --root $(MODEL_DIR)
	@touch $(MODEL_MARKER)

#################################################
### Cloud Run
#################################################

cloud_run_build: get_deploy_model
	@: $(eval VERSION := $(shell bash -c 'read -p "What version should it be tagged? " version; echo $$version'))
	@echo "${GCR_TAG}${VERSION}"
	@cd mush_app && gcloud builds submit --tag "${GCR_TAG}:${VERSION}"

cloud_run_deploy: cloud_run_build
	@cd mush_app && gcloud run deploy mushroom-classifier --image=${GCR_TAG}:latest --max-instances=1 --min-instances=0 --port=8080 \
--allow-unauthenticated --region=us-central1 --memory=16Gi --cpu=4 -q

cloud_run_make_public:
	@gcloud run services add-iam-policy-binding mushroom-classifier --member="allUsers" --role="roles/run.invoker"

cloud_run_delete:
	@gcloud run services delete mushroom-classifier --region=us-central1 -q

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
	@echo "dotenv:            					- Create the .env files"
	@echo "init:              					- Initialize the environment"
	@echo "all_datasets:      					- Download all dataset years"
	@echo "datasets_of_interest: 				- Download only the years of interest"
	@echo "fgvcx_2018:        					- Download the 2018 dataset"
	@echo "fgvcx_2019:        					- Download the 2019 dataset"
	@echo "fgvcx_2021:        					- Download the 2021 dataset"
	@echo "preprocess_data:   					- Preprocess the data & tidy unused files"
	@echo "tfrecords:         					- Convert the images to tfrecords with many user options"	
	@echo "download_model_weights: 				- Download Tensorflow model weights from a GitHub repo"
	@echo "get_deploy_model:  					- Download the latest registered model from wandb"
	@echo "deploy:            					- Deploy the model to a local server using ngrok"
	@echo "cloud_run_build:     					- Build the model on GCP using Cloud Build"
	@echo "cloud_run_deploy:  					- Deploy the model to Cloud Run"
	@echo "cloud_run_make_public: 				- Make the Cloud Run model public"
	@echo "cloud_run_delete:  					- Delete the Cloud Run model"
	@echo "terraform:         					- Deploy the model to GCP using Terraform"
	@echo "========================================="

#################################################
