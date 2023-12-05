# SMS based Mushroom Classifier <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [About the Project](#about-the-project)
- [TODO List](#todo-list)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Important Make Commands](#important-make-commands)

## About the Project
Hunting for mushrooms and foraging in general has become much more popular over the last few years. Unfortunately, people still make deadly misidentifications. This classifier does not aim to replace diligent identification, but to be used as a tool to put you in the right direction. Contained in this repo is all the code required to replicate the app yourself, including downloading the data, training and deployment. The app is deployed using Twilio webhooks to reply to MMS messages with an SMS containing the top 1-3 predictions based on how confident the model is. The training protocol results in top1 and top3 validation accuracies of 93% and 98%, respectively, for 467 species over 100 000 images of resolution (224, 224).

## TODO List

- [x] train model to an acceptable accuracy
- [x] create working docker image
- [x] implement fastapi in docker image and deploy on Google Run
- [x] logic to save hash of phone number and received image in BigQuery
- [ ] setup monitoring
- [ ] make a CI/CD pipeline with parameterized env vars
- [ ] create tests
- [ ] web scraper for MO images
- [ ] implement terraform?

## Getting Started

### Prerequisites

1. Ensure you have a working python 3.11 installation
2. Create a Twilio account and buy a phone number
3. Make a Google Cloud account & save the main service account credentials to the root of the repo
<!-- - Create two other service accounts and save the credentials: 
  - the first one will need admin privileges (Terraform)
  - the second will be for managing permissions around the files so leave it blank for now -->
4. Install Poetry
5. Set up a Weights and Biases account for MLOPs
6. Create an account with ngrok for local testing

### Installation

1. Clone this repository:

    `git clone https://github.com/Graham-Broughton/Mushroom-Classifier.git`

2. Create an .env file in the root directory using the .envsample file as a template
<!-- 3. Move the Terraform JSON credentials into the 'infrastructure' folder, rename it to terraform-account.json (the other ones can just stay in the root dir) -->

## Usage

As stated previously, this project was designed to be very easy to use while still permitting broad user configurability. In this section, we will cover the makefile commands provided and areas for user configurations.

### Important Make Commands

1. `make init` Runs make dotenv and build: copies the .env to needed directories and installs the required packages
2. `make -j2 datasets_of_interest` Download & extracts the required datasets (FGVCX 2018 & 2021). You will need around 700Gb of disk space for this step. The data strain is much lower when preprocessing is complete, so you can create a new VM with much less disk space afterwords.
3. `make tfrecords` Removes non-fungal images and processes and combines the associated json data from the datasets into a useable dataframe. This dataframe is then used for processing the images and important metadata into TFRecords using default resolution or your choice.
4. `make local_deploy` Download the latest model version from Weights and Biases and deploy it locally using ngrok.
5. `make cloud_run_build` Runs make get_deploy_model to download the latest model from wandb then builds a docker image with your chosen tag
6. `make cloud_run_deploy` Runs make cloud_run_build to package the app then deploys it on Google Cloud Run
7. `make cloud_run_delete` Deletes the deployed service

