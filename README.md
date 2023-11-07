# Mushroom Classifier

## An SMS based app by Graham Broughton

## About the Project
This repository contains all the code required to train and deploy a Swin (Shifted WINdows) transformer for mushroom classification. The model was trained on ~90 000 images containing ~470 species which resulted in ~80% top1 and ~96% top3 validation accuracy on images of resolution (224, 224). Training is performed on Google Cloud Platform (GCP) TPU v3-8 VM's where epoch durations are just shy of one minute. To deploy the model as an SMS based service, Twilio webhooks were set up to automatically respond to the client's number with an SMS if the message contained at least one photo. This project was designed with ease of use in mind, Makefile's were relied upon heavily to simplify operation down to a single word. Along the same lines, Terraform was used to simplify and ensure the proper GCP resources are requisitioned.

## Getting Started
<!-- Most of the dependencies are found in Pipfile's throughout the project but there are still a few things you need to do first: -->
### Prerequisites:

- Ensure you have a working python 3.11 installation
- Create a Twilio account and buy a phone number
- Make a Google Cloud account & save the main service account credentials
- Create two other service accounts and save the credentials: 
  - the first one will need admin privileges (Terraform)
  - the second will be for managing permissions around the files so leave it blank for
- Install Poetry

### Installation:

1. Clone this repository:

    `git clone https://github.com/Graham-Broughton/Mushroom-Classifier`

2. Create an .env file in the root directory from the .envsample template provided
3. Move the Terraform JSON credentials into the 'infrastructure' folder, rename to terraform-account.json

## Usage

As stated previously, this project was designed to be very easy to use while still permitting broad user configurability. In this section, we will cover the makefile commands provided and area's for user configurations.

### Pre-Use Configuration

1. `make build` Installs the requirements in a virtual environment managed by Poetry
2. `make dotenvs` Adds to, and copies the .env file you created to all the places it needs to be

### Data Preprocessing

1. `make datasets_of_interest` Downloads the required datasets that contain GPS data (FGVCX 2018 & 2021) using the respective make commands. You will need around 350Gb of disk space for this step. The data strain is much lower when preprocessing is complete, you can create a new VM with much less disk space afterwords.
2. ``



## Mushroom Classification
This is a package containing all the necessary ingredients to train  and deploy a model, in this case EffecientNetv2B0. Functionality is split into directories: one for training (training) and the other for deployment (deployment). This image classifier attained a top1 accuracy of almost 70% and top3 of 97% on the inat dataset without using metadata and top1 and top3 of 40%, and 78% and the FGVC dataset. This tool is meant to be used as an adjunct for proper identification protocol, not a replacement. Considering the relatively high top3 accuracy, it should be very useful to beginners to identify the mushroom to family or genus level where they can further identify it with a dichotomous key.

Training requirements:
- Ownership of a Google Cloud Storage bucket
- Create a config.yaml file following the template

Deployment requirements:
- Ability to launch a VM if you require remote inference

For deployment, you will also need to create a config file with your choice of dataset and model path if you decided to train as well. Further, you will need some local images (JPEG) of fungi to upload to the server once the model is deployed to make predictions on. This process was made painless by having a drag-and-drop feature for uploading images instead of web scraping them or uploading a directory with a specific structure. Once you upload an image, the model with return the top three predictions as well as the percent confidence for each.

**Table 1:** Starting screen (left) and the drag and drop image upload feature (right).
| Starting Screen | Drag and Drop Feature |
| --- | --- |
| <p align="center"><img src="https://github.com/Graham-Broughton/Mushroom-Classifier/blob/images/github_images/Screenshot%20(105).png" width="60%"></p> | <p align="center"><img src="https://github.com/Graham-Broughton/Mushroom-Classifier/blob/images/github_images/Screenshot%20(106).png" width="60%"></p> |

**Table 2:** The rest of the webpage's screens from left to right including: image preview after uploading, dynamic loading page, results (correct and incorrect). For the incorrect classification, the first guess was not even in the correct genus but the other two are much closer with the third being correct.
| Image Preview | Loading Results | Correct Classification | Incorrect Classification |
| --- | --- | --- | --- |
| <img src="https://github.com/Graham-Broughton/Mushroom-Classifier/blob/images/github_images/Screenshot%20(107).png"> | <img src="https://github.com/Graham-Broughton/Mushroom-Classifier/blob/images/github_images/Screenshot%20(108).png"> | <img src="https://github.com/Graham-Broughton/Mushroom-Classifier/blob/images/github_images/Screenshot%20(110).png"> | <img src="https://github.com/Graham-Broughton/Mushroom-Classifier/blob/images/github_images/Screenshot%20(109).png"> |

To run the training .py file you will need to choose which training set to use - inat or FGVC. You will do so by creating a config file following the template, along with any other modifications to the training protocol you so desire. Due to the massive size of the datasets, the training protocol is meant to be run using a TPU. It would be well worth your time to utilize the TFRecord notebooks to convert the datasets into TFRecords and store them in your Google Cloud Storage bucket. You will need to move the datasets into the storage bucket anyway since Google Cloud's TPU require training data to be in one of thier buckets.

Happy Hunting!
