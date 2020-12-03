# COVID19_AI_classifier

A detector for covid-19 chest X-ray (CXR) images using [scikit-learn](https://scikit-learn.org/stable/), a Python module for machine learning.

The deep feature extractor was developed by Professor Juliano Foleiss and you can find it [here](https://github.com/julianofoleiss/deep_feature_extractor).

This software was developed for educational purposes in the discipline of artificial intelligence at Federal University Of Technology Paraná – Brazil.

## :information_source: How To Use

```bash
# Pre-processing of images
$ python preprocessing_images.py data

# Extracting features
$ cd deep_feature_extractor/ && bash run.sh

# Predicting
$ python predict.py 

