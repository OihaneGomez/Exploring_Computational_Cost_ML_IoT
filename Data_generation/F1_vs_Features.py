import os
os.environ['PYTHONHASHSEED'] = '1'
from numpy.random import seed
seed(1)
import random as rn
rn.seed(1)

import warnings
warnings.filterwarnings("ignore")

import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier 
from scipy.stats import kurtosis 
from scipy.stats import skew
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
import timeit
import gc
from IPython import get_ipython
from sklearn.pipeline import Pipeline       
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict


#------------------Change these variables:------------------
#Number of signal components 
#(3 for XYZ, 1 for X)
number_components = 3

#Best feature number 
#(Max 162 for XYZ, 54 for X)
best_features_number=4

#Average results for n_times
n_times = 2

#Dataset folder relative route
root = '../HMP_Dataset/' #Regular Sampling
#root = '../HMP_Dataset_Downsample/' #Downsample
#----------------------------------------------------------


#Create activities dictionary 
wrist_class = {'Brush_teeth':0, 
              'Climb_stairs':4, 
              'Comb_hair':2, 
              'Descend_stairs':3, 
              'Drink_glass':1, 
              'Eat_meat':5, 
              'Eat_soup':6, 
              'Getup_bed':7, 
              'Liedown_bed':8, 
              'Pour_water':9, 
              'Sitdown_chair':10, 
              'Standup_chair':11, 
              'Use_telephone':12, 
              'Walk':13
             }

#Binarization dictionary
wrist_class_binary = {'Other':0, 
               'Drink_glass':1, 
             }
                
                
#Activity label list
wrist_labels = sorted(wrist_class, key=lambda x: x[1])
wrist_labels_binary = sorted(wrist_class_binary, key=lambda x: x[1], reverse=True) 


#Funtion to import all the txt for every class, calculate its features and append it in a Matrix
def gather(class_dict):
    df = []
    for c in class_dict.keys():
        f = glob.glob(root + c + '/*') #Get all files 
        d = pd.DataFrame(reformat(f, cls=c)) #Pandas dataframe for reformat funtion features
        df.append(d)
    return pd.concat(df)


#Median Filter axiliar function (https://stackoverflow.com/questions/41851044/python-median-filter-for-1d-numpy-array)    
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n)) 



#Obtain Data values for every secuence/instance
def reformat(files, cls):
    big_list= []
    for f in files:
        #Read every txt file (Number of row value_x, Value_Y, Value_Z)
        data = pd.read_csv(f, sep=' ', header=None, names=['x', 'y', 'z']) 
       
        #Conversion from 0-63 to m/s^2
        df_x = -14.709 + (data.iloc[:,0:1]/63)*(2*14.709)
        df_y = -14.709 + (data.iloc[:,1:2]/63)*(2*14.709)
        df_z = -14.709 + (data.iloc[:,2:3]/63)*(2*14.709)
        
    
        #Median filtering

        x = np.median(strided_app(df_x.values.flatten(), 3,1),axis=1)
        y = np.median(strided_app(df_y.values.flatten(), 3,1),axis=1)
        z = np.median(strided_app(df_z.values.flatten(), 3,1),axis=1)
        
        df_x = pd.DataFrame(x, columns=['x'])
        df_y = pd.DataFrame(y, columns=['y'])
        df_z = pd.DataFrame(z, columns=['z'])
        
        data_all=pd.concat([df_x.reset_index(drop=True), df_y.reset_index(drop=True), df_z.reset_index(drop=True)], axis=1)
        
        #Select the number of components
        data_reduced = data_all.iloc[:,0:number_components]

        #Split selected component of every secuence in X segments
        split_index=5 #Number of segments
        data_split=np.array_split(data_reduced, split_index)
        
        appended_features=[]
        
        #Features calculation
        features_whole = pd.concat([data_reduced.mean(axis=0).rename(index=lambda x: 'mean' + '_' + x), 
                              #Dataframe 
                              data_reduced.std(axis=0).rename(index=lambda x: 'std' + '_' + x),
                              data_reduced.median(axis=0).rename(index=lambda x: 'median' + '_' + x), 
                              data_reduced.mad(axis=0).rename(index=lambda x: 'mad' + '_' + x), 
                              data_reduced.max(axis=0).rename(index=lambda x: 'max' + '_' + x),
                              data_reduced.kurtosis(axis=0).rename(index=lambda x: 'kur' + '_' + x),  
                              data_reduced.skew(axis=0).rename(index=lambda x: 'skw' + '_' + x), 
                              data_reduced.var(axis=0).rename(index=lambda x: 'var' + '_' + x),   
                              data_reduced.min(axis=0).rename(index=lambda x: 'min' + '_' + x)])
        
        for i in range(0, split_index):
            #Features for every segment of data
            features= pd.concat([data_split[i].mean(axis=0).rename(index=lambda x: 'mean' + '_' + x + '_' + str(i)), 
                              data_split[i].std(axis=0).rename(index=lambda x: 'std' + '_' + x + '_' + str(i)),
                              data_split[i].median(axis=0).rename(index=lambda x: 'median' + '_' + x + '_' + str(i)), 
                              data_split[i].mad(axis=0).rename(index=lambda x: 'mad' + '_' + x + '_' + str(i)), 
                              data_split[i].max(axis=0).rename(index=lambda x: 'max' + '_' + x + '_' + str(i)),
                              data_split[i].kurtosis(axis=0).rename(index=lambda x: 'kur' + '_' + x + '_' + str(i)),  
                              data_split[i].skew(axis=0).rename(index=lambda x: 'skw' + '_' + x + '_' + str(i)),
                              #data_split[i].sum(axis=0).rename(index=lambda x: 'sum' + '_' + x + '_' + str(i)), 
                              data_split[i].var(axis=0).rename(index=lambda x: 'var' + '_' + x + '_' + str(i)),
                              data_split[i].min(axis=0).rename(index=lambda x: 'min' + '_' + x + '_' + str(i))])
            appended_features.append(features)
        
        #Concat all obtained features
        appended_features_all = pd.concat([features_whole,appended_features[0], appended_features[1], appended_features[2],appended_features[3],appended_features[4]])      #                
        
        #Binarize detectiom
        if wrist_class[cls] != 1:
            wrist_class[cls] = 0 #Other classes than drink are considered as CLASS 0. Drink = CLASS 1
        
        #Access to dictionary class number and add it as a feature
        appended_features_all['Y'] = wrist_class[cls]
        big_list.append(appended_features_all)

    #Return table containing all rows for every class and feature colums 
    #(mean*3, sd*3, Max*3, Min*3, Y). Number of the row is manteined. (0~101)
    return big_list

#Process all the dataset for creating the models

#Feature table for all classes
wrist_df = gather(wrist_class)

#Extact number of the class (feature Y)
wrist_Y = np.asarray(wrist_df.Y)

#Extact rest of colums (features)
num_features=len(wrist_df.columns)-1
wrist_X = np.asarray(wrist_df.iloc[:,:num_features])

#Normalize data (0-1)
min_max_scaler = preprocessing.MinMaxScaler()
wrist_X= min_max_scaler.fit_transform(wrist_df.iloc[:,:num_features])
wrist_X_df = pd.DataFrame(wrist_X, columns=(wrist_df.iloc[:,:num_features]).columns)


f1_score_array_LG_pre_mean, f1_score_array_RF_pre_mean, f1_score_array_KNN_pre_mean,f1_score_array_NB_pre_mean, f1_score_array_SVM_pre_mean, f1_score_array_MLP_pre_mean, f1_score_array_DT_pre_mean = [], [], [], [], [], [], []
f1_score_array_LG_pre_std, f1_score_array_RF_pre_std, f1_score_array_KNN_pre_std, f1_score_array_NB_pre_std, f1_score_array_SVM_pre_std, f1_score_array_MLP_pre_std, f1_score_array_DT_pre_std = [], [], [], [], [], [], []

f1_score_array_LG_mean, f1_score_array_RF_mean, f1_score_array_KNN_mean, f1_score_array_NB_mean, f1_score_array_SVM_mean, f1_score_array_MLP_mean, f1_score_array_DT_mean = [], [], [], [], [], [], []
f1_score_array_LG_std, f1_score_array_RF_std, f1_score_array_KNN_std, f1_score_array_NB_std, f1_score_array_SVM_std, f1_score_array_MLP_std, f1_score_array_DT_std = [], [], [], [], [], [], []

alg_array_names  = ['LG', 'RF', 'KNN', 'NB','SVM', 'MLP', 'DT']

n=0

#Classifiers
alg_array = [LogisticRegression(), 
         RandomForestClassifier(verbose= 0,n_estimators= 100,random_state= n),
         KNeighborsClassifier(n_neighbors=3),
         GaussianNB(), 
         svm.SVC(kernel='linear', C=64.0, probability=True, random_state = n),
         MLPClassifier(hidden_layer_sizes=(16, 16, 16), max_iter=1000,random_state = n),
         tree.DecisionTreeClassifier(random_state=n)]
    
array_dict_pre_mean = {"LG": f1_score_array_LG_pre_mean, 
          "RF": f1_score_array_RF_pre_mean,
          "KNN": f1_score_array_KNN_pre_mean,
          "NB": f1_score_array_NB_pre_mean,
          "SVM": f1_score_array_SVM_pre_mean,
          "MLP": f1_score_array_MLP_pre_mean,
          "DT": f1_score_array_DT_pre_mean
         }

array_dict_pre_std = {"LG": f1_score_array_LG_pre_std, 
          "RF": f1_score_array_RF_pre_std,
          "KNN": f1_score_array_KNN_pre_std,
          "NB": f1_score_array_NB_pre_std,
          "SVM": f1_score_array_SVM_pre_std,
          "MLP": f1_score_array_MLP_pre_std,
          "DT": f1_score_array_DT_pre_std
         }

array_dict_mean = {"LG": f1_score_array_LG_mean, 
              "RF": f1_score_array_RF_mean,
              "KNN": f1_score_array_KNN_mean,
              "NB": f1_score_array_NB_mean,
              "SVM": f1_score_array_SVM_mean,
              "MLP": f1_score_array_MLP_mean,
              "DT": f1_score_array_DT_mean
             }
array_dict_std = {"LG": f1_score_array_LG_std, 
              "RF": f1_score_array_RF_std,
              "KNN": f1_score_array_KNN_std,
              "NB": f1_score_array_NB_std,
              "SVM": f1_score_array_SVM_std,
              "MLP": f1_score_array_MLP_std,
              "DT": f1_score_array_DT_std
             }


#Calculate F1 n_times for every subset of features
for i in range (0,(best_features_number)):
    n_features = i+1
    select_feature = SelectKBest(chi2, k=n_features)
    for item, item_name in zip(alg_array, alg_array_names): 
        for n in range(0,n_times):
          
            #5-Cross validation
            kf = StratifiedKFold(n_splits=5, shuffle=True,random_state=n)

            #Pipeline to integrate feature selection within the cross validation and apply it only to the training fold
            pipe = Pipeline([('Feature selection',select_feature), ('Algorithm',item)])
            
            #Obtain f1_score
            scores = cross_val_score(pipe, wrist_X, wrist_Y, cv =kf, scoring = 'f1_macro')

            #Average F1 values for n_times
            average_score = np.mean(scores)*100
            array_dict_pre_mean[item_name].append(average_score)

        array_dict_mean[item_name].append(np.mean(array_dict_pre_mean[item_name]))
        array_dict_std[item_name].append(np.std(array_dict_pre_mean[item_name]))

        array_dict_pre_mean[item_name]=[]
        array_dict_pre_std[item_name]=[]


#Save data in .txt file, mean values
get_ipython().run_cell_magic('capture', 'cap --no-stderr', 'print (array_dict_mean)')
with open('F1_values_'+str(number_components)+'Comp_'+str(best_features_number)+'Features_mean.txt', 'w') as f:
    f.write(cap.stdout)    

#Save data in .txt file, std values
get_ipython().run_cell_magic('capture', 'cap ', 'print(array_dict_std)')
with open('F1_values_'+str(number_components)+'Comp_'+str(best_features_number)+'Features_std.txt', 'w') as f:
    f.write(cap.stdout) 
