# MAGIC Gamma Telescope Data Analysis with TPOT

This project aims to automate the process of applying machine learning to the MAGIC Gamma Telescope dataset using TPOT (Tree-based Pipeline Optimization Tool). TPOT is an AutoML tool that simplifies the selection and application of machine learning models.

## Project Overview

The MAGIC (Major Atmospheric Gamma Imaging Cherenkov) Telescope dataset contains data intended for binary classification tasks. The objective is to distinguish between signals (gamma particles) and background (hadrons) based on characteristics observed by the telescope.

We use TPOT to automatically find the best machine learning pipeline for this classification task, including data preprocessing steps and model selection.

## Dataset

The dataset is structured with the following features:

- `fLength`: Continuous feature representing the length of the major axis of the ellipse.
- `fWidth`: Continuous feature representing the length of the minor axis of the ellipse.
- `fSize`: Continuous feature representing the 10-log of the sum of the content of all pixels.
- `fConc`: Continuous feature representing the concentration of luminosity towards the center.
- `fConc1`: Continuous feature representing the ratio of the sum of the two highest pixels over fSize.
- `fAsym`: Continuous feature representing the distance from the highest pixel to the center.
- `fM3Long`: Continuous feature representing the 3rd root of the third moment along the major axis.
- `fM3Trans`: Continuous feature representing the 3rd root of the third moment along the minor axis.
- `fAlpha`: Continuous feature representing the angle of the major axis with the vector pointing to the origin.
- `fDist`: Continuous feature representing the distance from the origin to the center of the ellipse.
- `Class`: Categorical feature representing the class label (`g` for gamma, `h` for hadron).

## Requirements

- Python 3.6+
- Pandas
- NumPy
- Scikit-learn
- TPOT




