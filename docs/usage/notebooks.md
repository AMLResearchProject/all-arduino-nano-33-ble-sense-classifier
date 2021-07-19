# Notebooks Usage

![ALL Arduino Nano 33 BLE Sense Classifier](../img/project-banner.jpg)

# Introduction
This guide will take you through using the **ALL Arduino Nano 33 BLE Sense Classifier** Jupyter Notebook to train and test your classifier.

&nbsp;

# Installation
First you need to install the required software for training the model. Below are the available installation guides:

- [Ubuntu installation guide](../installation/ubuntu.md).

&nbsp;

# Network Architecture
We will build a Convolutional Neural Network with the following architecture:

- Average pooling layer
- Conv layer
- Depthwise conv layer
- Flatten layer
- Fully connected layer
- Softmax layer

&nbsp;

# Data
You need to be granted access to use the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. You can find the application form and information about getting access to the dataset on [this page](https://homes.di.unimi.it/scotti/all/#download) as well as information on how to contribute back to the project [here](https://homes.di.unimi.it/scotti/all/results.php).

_If you are not able to obtain a copy of the dataset please feel free to try this tutorial on your own dataset._

Once you have your data you need to add it to the project filesystem. You will notice the data folder in the Model directory, **model/data**, inside you have **train** & **test**. Add all of the images from the ALL_IDB1 dataset to the **model/data/train** folder.

## Data Augmentation

We will create an augmented dataset based on the [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon.

## Application testing data

In the data processing stage, ten negative images and ten positive images are removed from the dataset and moved to the **model/data/test/** directory. This data is not seen by the network during the training process, and is used to test the performance of the model.

To ensure your model gets the same results, you should use the same test images. You can also try with your own image selection, however results may vary.

&nbsp;

# Start Jupyter Notebooks
Now you need to start Jupyter Notebooks. In your project root execute the following command, replacing the IP and port as desired.

``` bash
jupyter notebook --ip YourIP --port 8888
```

&nbsp;

# Open The Training Notebook

Navigate to the URL provided when starting Jupyter Notebooks and you should be in the project root. Now navigate to **notebooks/classifier.ipynb**. With everything set up you can now begin training. Run the Jupyter Notebook and wait for it to finish.

&nbsp;

# Preparing For Arduino

During training the model was converted to TFLite and optimized with full integer quantization. The TFLite model was then converted to C array ready to be deployed to our Arduino Nano 33 BLE Sense. The test data that was removed before training was converted to 100px x 100px so as not to require additional resizing on the Arduino.

&nbsp;

# Conclusion

Here we trained a deep learning model for Acute Lymphoblastic Leukemia detection utilizing Intel® Optimization for Tensorflow* from the Intel® oneAPI AI Analytics Toolkit to optimize and accelarate training. We introduced a 6 layer deep learning model and applied data augmentation to increase the training data.

We trained our model with a target of 150 epochs and used early stopping to avoid overfitting. The model trained for 27 epochs resulting in a fairly good fit, and accuracy/precision/recall and AUC are satisfying. In addition the model reacts well during testing classifying each of the twenty unseen test images correctly.

&nbsp;

# Continue

Now you are ready to set up your Arduino Nano 33 BLE Sense. Head over to the [Arduino Installation Guide](../installation/arduino.md) to prepare your Arduino.

&nbsp;

# Contributing
Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss encourages and welcomes code contributions, bug fixes and enhancements from the Github community.

Please read the [CONTRIBUTING](https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier/blob/main/CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find our code of conduct in the [Code of Conduct](https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier/blob/main/CODE-OF-CONDUCT.md) document.

## Contributors
- [Adam Milton-Barker](https://www.leukemiaairesearch.com/association/volunteers/adam-milton-barker "Adam Milton-Barker") - [Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss") President/Founder & Lead Developer, Sabadell, Spain

&nbsp;

# Versioning
We use [SemVer](https://semver.org/) for versioning.

&nbsp;

# License
This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier/blob/main/LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues
We use the [repo issues](https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier/issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier/blob/main/CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.