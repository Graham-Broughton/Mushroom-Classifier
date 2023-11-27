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
Hunting for mushrooms and foraging in general has become much more popular over the last few years. Unfortunately, people still make deadly misidentifications. This classifier does not aim to replace diligent identification yourself, but as a tool to put you in the right direction. Contained in this repo is all the code required to replicate the app yourself, including downloading the data and training. The app is deployed using Twilio webhooks to reply to MMS messages with an SMS containing the top 1-3 predictions based on how confident the model is. The training protocol results in top1 and top3 validation accuracies of 80%  and 95%, respectively, for images of resolution (224, 224) over 467 species.

## TODO List

- [x] train model to acceptable accuracy
- [ ] create tests
- [ ] implement terraform?
- [ ] make a database for user images and ID
- [ ] web scraper for MO images

## Getting Started

### Prerequisites

- Ensure you have a working python 3.11 installation
- Create a Twilio account and buy a phone number
- Make a Google Cloud account & save the main service account credentials
- Create two other service accounts and save the credentials: 
  - the first one will need admin privileges (Terraform)
  - the second will be for managing permissions around the files so leave it blank for
- Install Poetry & set up a Weights and Biases account for MLOPs

### Installation

1. Clone this repository:

    `git clone https://github.com/Graham-Broughton/Mushroom-Classifier`

2. Create an .env file in the root directory using the .envsample file as a template
3. Move the Terraform JSON credentials into the 'infrastructure' folder, rename it to terraform-account.json (the other ones can just stay in the root dir)

## Usage

As stated previously, this project was designed to be very easy to use while still permitting broad user configurability. In this section, we will cover the makefile commands provided and areas for user configurations.

### Important Make Commands

1. `make init` Installs the requirements in a virtual environment managed by Poetry & copies the .env file to needed locations.
2. `make dotenv` Adds to, and copies the .env file you created to all the places it needs to be.
3. `make -j2 datasets_of_interest` Download & extracts the required datasets (FGVCX 2018 & 2021). You will need around 700Gb of disk space for this step. The data strain is much lower when preprocessing is complete, you can create a new VM with much less disk space afterwords.
4. `make tfrecords` Removes non-fungal images and processes and combines the associated json data from the datasets into a useable dataframe. This dataframe is then used for processing the images and important metadata into TFRecords.
5. `make deploy` Download the latest model version from Weights and Biases and deploy it.

