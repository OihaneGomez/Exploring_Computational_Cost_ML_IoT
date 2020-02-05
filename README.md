# Exploring_Computational_Cost_ML_IoT

## Exploring the Computational Cost of Machine Learning at the Edge for Human-Centric Internet of Things

Repository containing the material for the analysis of the paper "Exploring the Computational Cost of Machine Learning at the Edge for Human-Centric Internet of Things".

### Dataset
* [Dataset for ADL Recognition with Wrist-worn Accelerometer Data Set ](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer)

This data set contains 14 Activities of Daily Living (ADL) recordings of simple activities such as brushing teeth, climbing stairs, and drinking water from 16 volunteer participants. The data was recorded through a right-wrist worn tri-axial accelerometer on each of the participants.

### Structure of this repository

* `~/HMP_Dataset`    : Original dataset (32 Hz)
* `~/HMP_Dataset_Downsample`  : Downsampled dataset (16 Hz)
* `~/Data_generation`	: Python scripts for obtaining the results of the experiments
* `~/Visualisation`  : Jupyter notebooks for the representation of the results


### Reproducing the experiments

1. Install the dependencies `pip install -r requirements.txt`
2. Run the selected script included on `~/Python_scrips` to obtain the desired data (see *Notes* below)
3. Use the generated .txt files to plot the results with the Jupyter Notebooks incluided in `~/Jupyter_files`.



### Notes

All the experiments were done for the referenced dataset and the scripts are specifically created to process it. To use a different dataset it is necesarry to adapt the scripts and dataset. 

In the very first lines of some of the script we have included several variables:

* `root` determines the dataset folder, `HMP_Dataset` for the regular sampling and `HMP_Dataset_downsample` for the downsampled version
* `best_features_number` allows to select the number of features to conssider from the 162 initial subset
* `number_components` selects the number of signal components to consider, 1 for X and 3 for XYZ (this is specific for the evaluated use case)
* `n_times` Represents the number of times the cross validation process is performed to obtain the average f1 results
