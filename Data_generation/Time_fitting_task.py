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



#------------------Change these variables:------------------
#Number of signal components 
#(3 for XYZ, 1 for X)
number_components = 3

#Best feature number 
#(Max 162 for XYZ, 54 for X)
best_features_number=162

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
print (wrist_labels_binary)



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

#Total number of features
num_features=len(wrist_df.columns)-1

#Extact rest of colums (features)
wrist_X = np.asarray(wrist_df.iloc[:,:num_features])

#Normalize data (0-1)
min_max_scaler = preprocessing.MinMaxScaler()
wrist_X= min_max_scaler.fit_transform(wrist_df.iloc[:,:num_features])
wrist_X_df = pd.DataFrame(wrist_X, columns=(wrist_df.iloc[:,:num_features]).columns)


f1_score_array_LG_pre, f1_score_array_RF_pre, f1_score_array_KNN_pre, f1_score_array_NB_pre, f1_score_array_SVM_pre, f1_score_array_MLP_pre, f1_score_array_DT_pre = [], [], [], [], [], [], []
f1_score_array_LG, f1_score_array_RF, f1_score_array_KNN, f1_score_array_NB, f1_score_array_SVM, f1_score_array_MLP, f1_score_array_DT = [], [], [], [], [], [], []

f1_score_array_LG_pre_std, f1_score_array_RF_pre_std, f1_score_array_KNN_pre_std, f1_score_array_NB_pre_std, f1_score_array_SVM_pre_std, f1_score_array_MLP_pre_std, f1_score_array_DT_pre_std = [], [], [], [], [], [], []
f1_score_array_LG_std, f1_score_array_RF_std, f1_score_array_KNN_std, f1_score_array_NB_std, f1_score_array_SVM_std, f1_score_array_MLP_std, f1_score_array_DT_std = [], [], [], [], [], [], []


recall_scores, precision_scores, accuracy_scores, f1_scores = [],[],[],[]


for i in range (0,(best_features_number)):  
    best_features_number=i+1
    print( "Number of features: "+ str(best_features_number))
    select_feature = SelectKBest(chi2, k=best_features_number)
    
    #Create an array with the most representative features
    wrist_X_new = select_feature.fit_transform(wrist_X, wrist_Y)

    #Classifiers
    alg_array = [LogisticRegression(), 
                 RandomForestClassifier(verbose= 0,n_estimators= 100,random_state= n),
                 KNeighborsClassifier(n_neighbors=3),
                 GaussianNB(), 
                 svm.SVC(kernel='linear', C=64.0, probability=True, random_state = n),
                 MLPClassifier(hidden_layer_sizes=(16, 16, 16), max_iter=1000,random_state = n),
                 tree.DecisionTreeClassifier(random_state=n)]

    alg_array_names  = ['LG', 'RF', 'KNN', 'NB','SVM', 'MLP', 'DT']

    scoring = {'F1_score': 'f1_macro'}

    time_dict = {"LG": f1_score_array_LG, 
                  "RF": f1_score_array_RF,
                  "KNN": f1_score_array_KNN,
                  "NB": f1_score_array_NB,
                  "SVM": f1_score_array_SVM,
                  "MLP": f1_score_array_MLP,
                  "DT": f1_score_array_DT
                 }

    array_dict_std = {"LG": f1_score_array_LG_std, 
              "RF": f1_score_array_RF_std,
              "KNN": f1_score_array_KNN_std,
              "NB": f1_score_array_NB_std,
              "SVM": f1_score_array_SVM_std,
              "MLP": f1_score_array_MLP_std,
              "DT": f1_score_array_DT_std
             }


    for item, item_name in zip(alg_array, alg_array_names): 
        classifier = item

        #Calculate fitting time for every algorthm
        timer = get_ipython().run_line_magic('timeit', '-n 3 -r 100 -o classifier.fit(wrist_X_new, wrist_Y)')
        
        #Obtain average and standard deviation values
        timer_avg = (np.mean(timer.timings)*1000)
        time_std=(np.std(timer.timings)*1000)
        time_dict[item_name].append(timer_avg)
        array_dict_std[item_name].append(time_std)


