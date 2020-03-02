# Exploring_Computational_Cost_ML_IoT

## Exploring the Computational Cost of Machine Learning at the Edge for Human-Centric Internet of Things

Repository containing the material for the analysis of the paper "Exploring the Computational Cost of Machine Learning at the Edge for Human-Centric Internet of Things".

### Dataset
* [Dataset for ADL Recognition with Wrist-worn Accelerometer Data Set ](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer)

This data set contains 14 Activities of Daily Living (ADL) recordings of simple activities such as brushing teeth, climbing stairs, and drinking water from 16 volunteer participants. The data was recorded through a right-wrist worn tri-axial accelerometer on each of the participants.

### Structure of this repository

* `~/HMP_Dataset`    : Original dataset (32 Hz)
* `~/HMP_Dataset_Downsample`  : Downsampled dataset (16 Hz)
* `~/Data_generation`: Python scripts for obtaining the results of the experiments
    * `F1_vs_Features.py`   

         Save the mean and standard deviation values of the f1_macro score for every number of features.
    * `Time_data_processing_Inference.py` :  
  
         Save data processing times for inference, it will generate a .txt file containing average and std values 
    * `Time_data_processing_Train.py` : 
  
         Save data processing times for training, it will generate a .txt file containing average and std values 
    * `Time_fitting_task.py` : 
  
         Save fitting task times, it will generate a .txt file containing average and std values 
    * `Time_prediction_task.py` : 
  
         Save prediction task times, it will generate a .txt file containing average and std values 

* `~/Visualisation`  : Jupyter notebooks for the representation of the results
    * `Classification_results.ipynb` : 

        Obtain and/or visualize: Features ranking ,F1 vs Features plot (from saved file), Recall, precision, F1 and accuracy metrics 
    * `Time_results_prediction.ipynb` :   
 
        Visualize: Data Processing Times Vs Number of features, Prediction task times Vs Number of features   
    * `Time_results_training.ipynb` :  

        Visualize: Data Processing Times Vs Number of features, Fitting task times Vs Number of features   

### Reproducing the experiments

1. Install the dependencies `pip install -r requirements.txt`
2. Run the selected script included on `~/Python_scrips` to obtain the desired data (see *Notes* below). 
3. Use the generated .txt files to plot the results with the Jupyter Notebooks included in `~/Jupyter_files`.



### Notes

All the experiments were done for the referenced dataset and the scripts are specifically created to process it. To use a different dataset it is necessary to adapt the scripts and dataset. 

`Time_` staring scrips must be run using `ipython3`. Example: `ipython3 Time_data_processing_Inference.py` 


In the very first lines of some of the script we have included several variables:

* `root` determines the dataset folder, `HMP_Dataset` for the regular sampling and `HMP_Dataset_downsample` for the downsampled version
* `best_features_number` allows selecting the number of features to consider from the 162 initial subset
* `number_components` selects the number of signal components to consider, 1 for X and 3 for XYZ (this is specific for the evaluated use case)
* `n_times` Represents the number of times the cross-validation process is performed to obtain the average f1 results
