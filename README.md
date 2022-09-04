# Mushroom Classification
This is a package containing all the necassary ingredients to train and deploy a model, in this case EffecientNetv2B0. It has two packages: one for training (training) and the other for deployment (deployment). 

Training requirements:
- Ownership of a Google Cloud Storage bucket
- Create a config.yaml file following the template

Deployment requirements:
- Ability to launch a VM if you require remote inference

For deployment, you will also need to create a config file with your choice of dataset and model path if you decided to train as well. Further, you will need some local images (JPEG) of fungi to upload to the server once the model is deployed to make predictions on. This process was made painless by having a drag-and-drop feature for uploading images instead of web scraping them or uploading a directory with a specific structure. Once you upload an image, the model with return the top three predictions as well as the percent confidence for each.

<p align="center" width="100%">
    <img width="60%" src="github_images/Screenshot (105).png"> 
</p>

<table>
  <tr>
    <td><img src="github_images/Screenshot (107).png"></td>
    <td><img src="github_images/Screenshot (108).png"></td>
    <td><img src="github_images/Screenshot (109).png"></td>
    <td><img src="github_images/Screenshot (110).png"></td>
  </tr>
</table>

<table>
    <tr>
        <td><img src="github_images/Screenshot (105).png"></td>
    </tr>
    <tr>
        <td><img src="github_images/Screenshot (106).png"></td>
    </tr>
</table>

To run the training .py file you will need to choose which training set to use - inat or FGVC. You will do so by creating a config file following the template, along with any other modifications to the training protocol you so desire. Due to the massive size of the datasets, the training protocol is meant to be run using a TPU. It would be well worth your time to utilize the TFRecord notebooks to convert the datasets into TFRecords and store them in your Google Cloud Storage bucket. You will need to move the datasets into the storage bucket anyway since Google Cloud's TPU require training data to be in one of thier buckets.

I learned a ton making this project, I hope you can learn from it too!
