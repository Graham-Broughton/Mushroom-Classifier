# Mushroom Classification
This is a package containing all the necassary ingredients to train and deploy a model, in this case EffecientNetv2B0. Functionality is split between bracnhes: one for training (training) and the other for deployment (deployment). This image classifier attained a top1 accuracy of almost 70% and top3 of 97% on the inat dataset without using metadata and top1 and top3 of 40%, and 78% and the FGVC dataset. This tool is meant to be used as an adjunct for proper identification protocol, not a replacement. Considering the relatively high top3 accuracy, it should be very useful to beginners to identify the mushroom to family or genus level where they can further identify it with a dichotomous key.

Training requirements:
- Ownership of a Google Cloud Storage bucket
- Create a config.yaml file following the template

Deployment requirements:
- Ability to launch a VM if you require remote inference

For deployment, you will also need to create a config file with your choice of dataset and model path if you decided to train as well. Further, you will need some local images (JPEG) of fungi to upload to the server once the model is deployed to make predictions on. This process was made painless by having a drag-and-drop feature for uploading images instead of web scraping them or uploading a directory with a specific structure. Once you upload an image, the model with return the top three predictions as well as the percent confidence for each.

**Table 1:** Starting screen (left) and the drag and drop image upload feature (right).
| Starting Screen | Drag and Drop Feature |
| --- | --- |
| <p align="center"><img src="github_images/Screenshot (105).png" width="60%"></p> | <p align="center"><img src="github_images/Screenshot (106).png" width="60%"></p> |

**Table 2:** The rest of the webpage's screens from left to right including: image preview after uploading, dynamic loading page, results (correct and incorrect). For the incorrect classification, the first guess was not even in the correct genus but the other two are much closer with the third being correct.
| Image Preview | Loading Results | Correct Classification | Incorrect Classification |
| --- | --- | --- | --- |
| <img src="github_images/Screenshot (107).png"> | <img src="github_images/Screenshot (108).png"> | <img src="github_images/Screenshot (110).png"> | <img src="github_images/Screenshot (109).png"> |

To run the training .py file you will need to choose which training set to use - inat or FGVC. You will do so by creating a config file following the template, along with any other modifications to the training protocol you so desire. Due to the massive size of the datasets, the training protocol is meant to be run using a TPU. It would be well worth your time to utilize the TFRecord notebooks to convert the datasets into TFRecords and store them in your Google Cloud Storage bucket. You will need to move the datasets into the storage bucket anyway since Google Cloud's TPU require training data to be in one of thier buckets.

Happy Hunting!
