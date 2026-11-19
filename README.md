# PEC-CL-SU-Classification

Overview

This repository contains the Python scripts used for machine-learning-based classification of five structurally similar sulfonylurea herbicides (SUs) in the study entitled "Decoupling Analyte Identity from Concentration by Multidimensional Photoelectrochemical–Colorimetric Fingerprinting".

Contents

The repository includes the following scripts:

RF_classification.py: Random forest classification and confusion matrix generation (Figure 4D).

GNB_classification.py: Gaussian Naive Bayes classification.

LDA_classification.py: Linear discriminant analysis.

SVM_classification.py: Support vector machine classification.

KNN_classification.py: K-nearest neighbors classification.

The scripts for model comparison and confusion matrix visualization correspond to Figures 4E and S8, respectively.

Software and Dependencies

The analysis was performed using Python and the scikit-learn library. The required packages include:

pandas

numpy

scikit-learn

matplotlib

Usage

Download the required Python script and prepare the input CSV file according to the specified format. Run the corresponding script in a Python environment with the required dependencies installed.
