# Ubuntu Installation

![ALL Arduino Nano 33 BLE Sense Classifier](../img/project-banner.jpg)

# Introduction
This guide will take you through the installation process for the **ALL Arduino Nano 33 BLE Sense Classifier** trainer.

&nbsp;

# Operating System
This project supports the following operating system(s), but may work as described on other OS.

- [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)

&nbsp;

# Software
This project uses the following libraries.

- Conda
- Intel® oneAPI AI Analytics Toolkit
- Jupyter Notebooks
- NBConda
- Mlxtend
- Pillow
- Opencv
- Scipy
- Scikit Image
- Scikit Learn

&nbsp;

# Clone the repository

Clone the [ALL Arduino Nano 33 BLE Sense Classifier](https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier " ALL Arduino Nano 33 BLE Sense Classifier") repository from the [Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project](https://github.com/AMLResearchProject "Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project") Github Organization.

To clone the repository and install the project, make sure you have Git installed. Now navigate to the directory you would like to clone the project to and then use the following command.

``` bash
 git clone https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier.git
```

This will clone the ALL Arduino Nano 33 BLE Sense Classifier repository.

``` bash
 ls
```

Using the ls command in your home directory should show you the following.

``` bash
 ALL-Arduino-Nano-33-BLE-Sense-Classifier
```

Navigate to the **ALL-Arduino-Nano-33-BLE-Sense-Classifier** directory, this is your project root directory for this tutorial.

## Developer forks

Developers from the Github community that would like to contribute to the development of this project should first create a fork, and clone that repository. For detailed information please view the [CONTRIBUTING](https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier/blob/master/CONTRIBUTING.md "CONTRIBUTING") guide. You should pull the latest code from the development branch.

``` bash
 git clone -b "2.0.0" https://github.com/AMLResearchProject/ALL-Arduino-Nano-33-BLE-Sense-Classifier.git
```

The **-b "2.0.0"** parameter ensures you get the code from the latest master branch. Before using the below command please check our latest master branch in the button at the top of the project README.

&nbsp;

# Installation
You are now ready to install the ALL Arduino Nano 33 BLE Sense Classifier trainer. All software requirements are included in **scripts/install.sh**. You can run this file on your machine from the project root in terminal. Use the following command:

``` bash
 sh scripts/install.sh
```

**WARNING:** This script assumes you have not already installed the oneAPI Basekit.

**WARNING:** This script assumes you have not already installed the oneAPI AI Analytics Toolkit.

**WARNING:** This script assumes you have an Intel GPU.

**WARNING:** This script assumes you have already installed the Intel GPU drivers.

**HINT:** If any of the above are not relevant to you, please comment out the relevant sections below before running this installation script.

&nbsp;

# Continue

Choose one of the following usage guides to train your model:

- [Python Usage Guide](../usage/python.md)
- [Jupyter Notebooks Usage Guide](../usage/notebooks.md)

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