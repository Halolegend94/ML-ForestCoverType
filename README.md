# ML Project: Forest Cover Type Classification

The objective of this work is to develop and compare different models of a phenomenon using machine learning techniques. In particular, the interest lays in predicting the forest cover type of a region given certain characteristics of the soil and ambient.

In order to do so, large dataset of 581012 instances is provided, where each instance is an observation of a 30 X 30 meter cell characterized by 54 attributes and associated with the correct class of forest cover type. The 54 columns of data are divided as follows: 10 quantitative variables, 4 binary wilderness areas and 40 binary soil type variables.

The complete report is <a href="Report/main.pdf">HERE</a>

## Repository Structure

* `Libreria` contains a bunch of ML utilities written by me. For example `Dataset` class helps in handling dataset splits and other kinds of manipulation.
* `Report` contains the report latex source and the compiled pdf.
* `Outputs` contains text files in which there are scripts'output for each classifier.
* `[classifier].py` are the scripts that test various classifiers.
* `splitDataset.py` is the script used to divide the main dataset in K folds (I used a precomputed split for comparison reasons)
* `TestBase.py` is a module that contains a function enclosing the tests a classifier is subject to. It is imported and
   used in other scripts.
* `DataHandler.py` contains the function that loads data in memory.

## Get the data

You can obtain the dataset used <a href="https://archive.ics.uci.edu/ml/datasets/Covertype">here</a>. 
