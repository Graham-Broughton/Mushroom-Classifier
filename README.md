# Mushroom Classification
This is a package containing all the necassary ingredients to train and deploy a model, in this case EffecientNetv2B0. It has two packages: one for training (training) and the other for deployment (deployment). To run the training .py file you will need to choose which training set to use - inat or FGVC. You will do so by creating a config file following the example, along with any other modifications to the training protocol you so desire. 

The training protocol is meant to be run using a TPU, so it would be very beneficial for your training time to utilize the TFRecord notebooks to convert the datasets into TFRecords and store them in your Google Cloud Storage bucket.

For deployment, you will also need to create a config file with the saed model path. Further, you will need loval images of fungi to upload once the model is deployed. It has a easy to use drag-and-drop feature for uploading images for testing. The model produces the top three predictions as well as it's percent confidence.

I hope this package may help you.
