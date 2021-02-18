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
import multiprocess as mp
from functools import partial
import functools


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


process_pool = mp.Pool(processes=mp.cpu_count())

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
wrist_labels_binary = sorted(wrist_class_binary, key=lambda x: x[1]) 



#Median Absolute Deviation
def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


#Meadian Filter
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))       



#Import all the txt for every class, calculate its features and append it in a matrix
def gather(class_dict,features):
    df = []
    data2=pd.DataFrame()
    for c in class_dict.keys():
        f = glob.glob(root + c + '/*') #Get all files 
        g = functools.partial(reformat,  cls=c, features=features)
        dfs = process_pool.map(g, f)
        data1 = pd.concat(dfs, ignore_index=True)
        data2=pd.concat([data1, data2], ignore_index=True)
    return data2

appended_features_df = pd.DataFrame()

#Import only one sequence per class (14 total)
def gather_one(class_dict, features):
    df = []
    for c in class_dict.keys():
        f = glob.glob(root + c + '/*') #Get all files 
        f_final=f[0:1]
        d = pd.DataFrame(reformat_one(f_final, cls=c, features=features)) #Pandas dataframe for reformat funtion features
        df.append(d)
    return pd.concat(df)

appended_features_df = pd.DataFrame()




#Obtain Data values for every secuence/instance
def reformat_one(files, cls, features):
    appended_features_all=[]
    appended_features_df = pd.DataFrame()
    for f in files:
        #Read every txt file (Number of row value_x, Value_Y, Value_Z)
        
        if number_components == 3: #FOR XYZ
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
                    
            data_x = df_x.values
            data_y = df_y.values
            data_z = df_z.values
            
            #Divide data in segments
            split_index=5 #Number of segments
            data_split_x=np.array_split(data_x, split_index)
            data_split_y=np.array_split(data_y, split_index)
            data_split_z=np.array_split(data_z, split_index)
            
            #Features Calculation

            appended_before=['data_split_x[2].min(axis=0)', 'data_x.min(axis=0)','data_split_x[2].mean(axis=0)',
            'np.median(data_split_x[2],axis=0)','data_split_x[1].min(axis=0)','data_split_x[1].mean(axis=0)',
            'data_x.mean(axis=0)','np.median(data_split_x[1],axis=0)','np.median(data_x,axis=0)',
            'data_split_x[3].mean(axis=0)','np.median(data_split_x[3],axis=0)','data_split_x[3].min(axis=0)',
            'np.median(data_z,axis=0)','data_split_x[4].min(axis=0)','data_split_x[2].max(axis=0)',
            'np.median(data_split_z[2],axis=0)','data_split_x[2].std(axis=0)','data_split_x[4].mean(axis=0)',
            'data_split_x[3].max(axis=0)','np.median(data_split_x[4],axis=0)','data_z.std(axis=0)',
            'mad(data_split_x[2],axis=0)','np.median(data_split_z[4],axis=0)','data_split_z[2].mean(axis=0)',
            'mad(data_z,axis=0)','data_split_z[2].std(axis=0)','data_z.mean(axis=0)',
            'data_split_z[4].mean(axis=0)','data_split_x[0].min(axis=0)','data_z.var(axis=0)',
            'np.median(data_split_z[3],axis=0)','data_split_z[3].mean(axis=0)','mad(data_split_z[2],axis=0)',
            'np.median(data_split_x[0],axis=0)','data_split_x[0].mean(axis=0)','data_split_x[1].max(axis=0)',
            'data_z.min(axis=0)','data_split_x[4].var(axis=0)','data_split_x[2].var(axis=0)',
            'data_split_z[2].var(axis=0)','data_split_z[1].std(axis=0)','data_split_z[2].min(axis=0)',
            'data_split_x[4].std(axis=0)','data_split_z[4].var(axis=0)','mad(data_split_z[1],axis=0)',
            'mad(data_split_y[3],axis=0)','mad(data_split_x[4],axis=0)','mad(data_y,axis=0)',
            'data_split_z[1].var(axis=0)','data_split_z[3].max(axis=0)','data_split_z[4].std(axis=0)',
            'mad(data_split_z[4],axis=0)','data_split_z[1].min(axis=0)','data_y.std(axis=0)',
            'data_split_y[3].std(axis=0)','data_split_z[4].max(axis=0)','data_split_z[0].min(axis=0)',
            'data_split_z[1].mean(axis=0)','data_split_x[0].var(axis=0)','data_split_z[3].min(axis=0)',
            'np.median(data_split_z[1],axis=0)','data_x.var(axis=0)','np.median(data_split_z[0],axis=0)',
            'data_split_z[4].min(axis=0)','data_y.var(axis=0)','data_split_z[0].mean(axis=0)',
            'data_split_x[0].std(axis=0)','kurtosis(data_split_z[4],axis=0)','np.median(data_split_y[2],axis=0)',
            'data_split_x[4].max(axis=0)','data_split_y[3].var(axis=0)','data_x.max(axis=0)',
            'data_split_z[0].var(axis=0)','data_split_y[2].max(axis=0)','data_split_y[2].mean(axis=0)',
            'mad(data_split_x[0],axis=0)','data_split_z[3].var(axis=0)','data_x.std(axis=0)',
            'kurtosis(data_split_y[1],axis=0)','data_split_z[0].std(axis=0)','data_split_z[2].max(axis=0)',
            'mad(data_split_z[0],axis=0)','kurtosis(data_y,axis=0)','data_split_y[0].min(axis=0)',
            'data_split_z[3].std(axis=0)','data_split_x[1].std(axis=0)','kurtosis(data_split_y[0],axis=0)',
            'skew(data_z,axis=0)','mad(data_split_z[3],axis=0)','skew(data_split_y[2],axis=0)',
            'data_split_x[1].var(axis=0)','data_split_x[0].max(axis=0)','np.median(data_split_y[4],axis=0)',
            'data_split_y[4].mean(axis=0)','mad(data_x,axis=0)','data_split_y[0].mean(axis=0)',
            'data_split_y[2].var(axis=0)','data_split_z[0].max(axis=0)','np.median(data_split_y[3],axis=0)',
            'data_split_z[1].max(axis=0)','data_split_y[2].std(axis=0)','data_split_y[3].max(axis=0)',
            'mad(data_split_x[1],axis=0)','np.median(data_split_y[0],axis=0)','mad(data_split_y[1],axis=0)',
            'data_split_y[3].mean(axis=0)','mad(data_split_y[2],axis=0)','data_split_y[0].max(axis=0)',
            'kurtosis(data_x,axis=0)','data_split_y[1].min(axis=0)','skew(data_split_y[3],axis=0)',
            'skew(data_split_x[3],axis=0)','kurtosis(data_split_y[3],axis=0)','data_split_y[4].min(axis=0)',
            'data_split_y[0].var(axis=0)','mad(data_split_x[3],axis=0)','data_split_y[1].std(axis=0)',
            'kurtosis(data_split_z[1],axis=0)','kurtosis(data_split_y[4],axis=0)','skew(data_split_z[2],axis=0)',
            'skew(data_split_x[1],axis=0)','data_split_y[4].max(axis=0)','np.median(data_y,axis=0)',
            'data_split_y[4].std(axis=0)','skew(data_split_z[1],axis=0)','kurtosis(data_split_x[2],axis=0)',
            'skew(data_split_x[2],axis=0)','data_split_y[1].mean(axis=0)','kurtosis(data_split_y[2],axis=0)',
            'skew(data_split_z[0],axis=0)','kurtosis(data_split_x[0],axis=0)','skew(data_split_y[0],axis=0)',
            'data_split_y[1].max(axis=0)','skew(data_split_z[3],axis=0)','kurtosis(data_split_x[1],axis=0)',
            'kurtosis(data_split_x[3],axis=0)','data_split_x[3].std(axis=0)','skew(data_y,axis=0)',
            'data_z.max(axis=0)','mad(data_split_y[4],axis=0)','data_y.mean(axis=0)',
            'np.median(data_split_y[1],axis=0)','data_y.max(axis=0)','skew(data_x,axis=0)',
            'data_split_y[4].var(axis=0)','mad(data_split_y[0],axis=0)','skew(data_split_y[1],axis=0)',
            'kurtosis(data_z,axis=0)','kurtosis(data_split_x[4],axis=0)','data_split_y[2].min(axis=0)',
            'kurtosis(data_split_z[2],axis=0)','skew(data_split_y[4],axis=0)','data_split_y[1].var(axis=0)',
            'data_split_x[3].var(axis=0)','kurtosis(data_split_z[0],axis=0)','data_split_y[3].min(axis=0)',
            'kurtosis(data_split_z[3],axis=0)','data_split_y[0].std(axis=0)','skew(data_split_x[4],axis=0)',
            'skew(data_split_x[0],axis=0)','skew(data_split_z[4],axis=0)','data_y.min(axis=0)']
        else: # For the most representative component (X)


            data = pd.read_csv(f, sep=' ', header=None, names=['x']) 
            
            #Conversion from 0-63 to m/s^2
            df_x = -14.709 + (data.iloc[:,0:1]/63)*(2*14.709)
            
            #Median filtering
            x = np.median(strided_app(df_x.values.flatten(), 3,1),axis=1)

            df_x = pd.DataFrame(x, columns=['x'])
  
            data_x = df_x.values

            #Divide data in segments
            split_index=5 #Number of segments
            data_split_x=np.array_split(data_x, split_index)
  
            appended_before=['data_split_x[2].min(axis=0)','data_x.min(axis=0)','data_split_x[2].mean(axis=0)',
            'np.median(data_split_x[2],axis=0)','data_split_x[1].min(axis=0)','data_split_x[1].mean(axis=0)',
            'data_x.mean(axis=0)','np.median(data_split_x[1],axis=0)','np.median(data_x,axis=0)',
            'data_split_x[3].mean(axis=0)','np.median(data_split_x[3],axis=0)','data_split_x[3].min(axis=0)',
            'data_split_x[4].min(axis=0)','data_split_x[2].max(axis=0)','data_split_x[2].std(axis=0)',
            'data_split_x[4].mean(axis=0)','data_split_x[3].max(axis=0)','np.median(data_split_x[4],axis=0)',
            'mad(data_split_x[2],axis=0)','data_split_x[0].min(axis=0)','np.median(data_split_x[0],axis=0)',
            'data_split_x[0].mean(axis=0)','data_split_x[1].max(axis=0)','data_split_x[4].var(axis=0)',
            'data_split_x[2].var(axis=0)','data_split_x[4].std(axis=0)','mad(data_split_x[4],axis=0)',
            'data_split_x[0].var(axis=0)','data_x.var(axis=0)','data_split_x[0].std(axis=0)',
            'data_split_x[4].max(axis=0)','data_x.std(axis=0)','mad(data_split_x[0],axis=0)',
            'data_split_x[1].std(axis=0)','data_x.max(axis=0)','data_split_x[1].var(axis=0)',
            'data_split_x[0].max(axis=0)','mad(data_x,axis=0)','kurtosis(data_x,axis=0)',
            'mad(data_split_x[1],axis=0)','skew(data_split_x[3],axis=0)','mad(data_split_x[3],axis=0)',
            'skew(data_split_x[1],axis=0)','kurtosis(data_split_x[2],axis=0)','skew(data_split_x[2],axis=0)',
            'skew(data_x,axis=0)','kurtosis(data_split_x[0],axis=0)','kurtosis(data_split_x[1],axis=0)',
            'kurtosis(data_split_x[3],axis=0)','data_split_x[3].std(axis=0)','kurtosis(data_split_x[4],axis=0)',
            'data_split_x[3].var(axis=0)','skew(data_split_x[4],axis=0)','skew(data_split_x[0],axis=0)']

        #Create initial_features_matrix
        appended_features_split=[]
        appended_features=[]

        for i in range (0, features):
            appended_features_before = eval(appended_before[i])
            appended_features.append(appended_features_before[0])
            appended_features_before=[] 

        appended_features_all.append(appended_features)


    appended_features_df = pd.DataFrame(appended_features_all)
           
    #Binarize detectiom
    if wrist_class[cls] != 1:
        wrist_class[cls] = 0 #Other classes than drink are considered as CLASS 0. Drink = CLASS 1
    
    #Access to dictionary class number and add it as a feature
    appended_features_df[-1]= wrist_class[cls]
    

    #Return table containing all rows for every class and feature colums 
    #(mean*3, sd*3, Max*3, Min*3, Y). Number of the row is manteined. (0~101)
    return appended_features_df



def reformat (files, features, cls):
    #Median Absolute Deviation
    def mad(data, axis=None):
        return np.mean(np.absolute(data - np.mean(data, axis)), axis)


    #Meadian Filter
    def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size-L)//S)+1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))       


    appended_features_all=[]
    appended_features_df = pd.DataFrame()
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


    if number_components == 3: #FOR XYZ
        data = pd.read_csv(files, sep=' ', header=None, names=['x', 'y', 'z']) 
        
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
        

        data_x = df_x.values
        data_y = df_y.values
        data_z = df_z.values
        
        #Divide data in segments
        split_index=5 #Number of segments
        data_split_x=np.array_split(data_x, split_index)
        data_split_y=np.array_split(data_y, split_index)
        data_split_z=np.array_split(data_z, split_index)
        
        #Features Calculation

        appended_before=['data_split_x[2].min(axis=0)', 'data_x.min(axis=0)','data_split_x[2].mean(axis=0)',
        'np.median(data_split_x[2],axis=0)','data_split_x[1].min(axis=0)','data_split_x[1].mean(axis=0)',
        'data_x.mean(axis=0)','np.median(data_split_x[1],axis=0)','np.median(data_x,axis=0)',
        'data_split_x[3].mean(axis=0)','np.median(data_split_x[3],axis=0)','data_split_x[3].min(axis=0)',
        'np.median(data_z,axis=0)','data_split_x[4].min(axis=0)','data_split_x[2].max(axis=0)',
        'np.median(data_split_z[2],axis=0)','data_split_x[2].std(axis=0)','data_split_x[4].mean(axis=0)',
        'data_split_x[3].max(axis=0)','np.median(data_split_x[4],axis=0)','data_z.std(axis=0)',
        'mad(data_split_x[2],axis=0)','np.median(data_split_z[4],axis=0)','data_split_z[2].mean(axis=0)',
        'mad(data_z,axis=0)','data_split_z[2].std(axis=0)','data_z.mean(axis=0)',
        'data_split_z[4].mean(axis=0)','data_split_x[0].min(axis=0)','data_z.var(axis=0)',
        'np.median(data_split_z[3],axis=0)','data_split_z[3].mean(axis=0)','mad(data_split_z[2],axis=0)',
        'np.median(data_split_x[0],axis=0)','data_split_x[0].mean(axis=0)','data_split_x[1].max(axis=0)',
        'data_z.min(axis=0)','data_split_x[4].var(axis=0)','data_split_x[2].var(axis=0)',
        'data_split_z[2].var(axis=0)','data_split_z[1].std(axis=0)','data_split_z[2].min(axis=0)',
        'data_split_x[4].std(axis=0)','data_split_z[4].var(axis=0)','mad(data_split_z[1],axis=0)',
        'mad(data_split_y[3],axis=0)','mad(data_split_x[4],axis=0)','mad(data_y,axis=0)',
        'data_split_z[1].var(axis=0)','data_split_z[3].max(axis=0)','data_split_z[4].std(axis=0)',
        'mad(data_split_z[4],axis=0)','data_split_z[1].min(axis=0)','data_y.std(axis=0)',
        'data_split_y[3].std(axis=0)','data_split_z[4].max(axis=0)','data_split_z[0].min(axis=0)',
        'data_split_z[1].mean(axis=0)','data_split_x[0].var(axis=0)','data_split_z[3].min(axis=0)',
        'np.median(data_split_z[1],axis=0)','data_x.var(axis=0)','np.median(data_split_z[0],axis=0)',
        'data_split_z[4].min(axis=0)','data_y.var(axis=0)','data_split_z[0].mean(axis=0)',
        'data_split_x[0].std(axis=0)','kurtosis(data_split_z[4],axis=0)','np.median(data_split_y[2],axis=0)',
        'data_split_x[4].max(axis=0)','data_split_y[3].var(axis=0)','data_x.max(axis=0)',
        'data_split_z[0].var(axis=0)','data_split_y[2].max(axis=0)','data_split_y[2].mean(axis=0)',
        'mad(data_split_x[0],axis=0)','data_split_z[3].var(axis=0)','data_x.std(axis=0)',
        'kurtosis(data_split_y[1],axis=0)','data_split_z[0].std(axis=0)','data_split_z[2].max(axis=0)',
        'mad(data_split_z[0],axis=0)','kurtosis(data_y,axis=0)','data_split_y[0].min(axis=0)',
        'data_split_z[3].std(axis=0)','data_split_x[1].std(axis=0)','kurtosis(data_split_y[0],axis=0)',
        'skew(data_z,axis=0)','mad(data_split_z[3],axis=0)','skew(data_split_y[2],axis=0)',
        'data_split_x[1].var(axis=0)','data_split_x[0].max(axis=0)','np.median(data_split_y[4],axis=0)',
        'data_split_y[4].mean(axis=0)','mad(data_x,axis=0)','data_split_y[0].mean(axis=0)',
        'data_split_y[2].var(axis=0)','data_split_z[0].max(axis=0)','np.median(data_split_y[3],axis=0)',
        'data_split_z[1].max(axis=0)','data_split_y[2].std(axis=0)','data_split_y[3].max(axis=0)',
        'mad(data_split_x[1],axis=0)','np.median(data_split_y[0],axis=0)','mad(data_split_y[1],axis=0)',
        'data_split_y[3].mean(axis=0)','mad(data_split_y[2],axis=0)','data_split_y[0].max(axis=0)',
        'kurtosis(data_x,axis=0)','data_split_y[1].min(axis=0)','skew(data_split_y[3],axis=0)',
        'skew(data_split_x[3],axis=0)','kurtosis(data_split_y[3],axis=0)','data_split_y[4].min(axis=0)',
        'data_split_y[0].var(axis=0)','mad(data_split_x[3],axis=0)','data_split_y[1].std(axis=0)',
        'kurtosis(data_split_z[1],axis=0)','kurtosis(data_split_y[4],axis=0)','skew(data_split_z[2],axis=0)',
        'skew(data_split_x[1],axis=0)','data_split_y[4].max(axis=0)','np.median(data_y,axis=0)',
        'data_split_y[4].std(axis=0)','skew(data_split_z[1],axis=0)','kurtosis(data_split_x[2],axis=0)',
        'skew(data_split_x[2],axis=0)','data_split_y[1].mean(axis=0)','kurtosis(data_split_y[2],axis=0)',
        'skew(data_split_z[0],axis=0)','kurtosis(data_split_x[0],axis=0)','skew(data_split_y[0],axis=0)',
        'data_split_y[1].max(axis=0)','skew(data_split_z[3],axis=0)','kurtosis(data_split_x[1],axis=0)',
        'kurtosis(data_split_x[3],axis=0)','data_split_x[3].std(axis=0)','skew(data_y,axis=0)',
        'data_z.max(axis=0)','mad(data_split_y[4],axis=0)','data_y.mean(axis=0)',
        'np.median(data_split_y[1],axis=0)','data_y.max(axis=0)','skew(data_x,axis=0)',
        'data_split_y[4].var(axis=0)','mad(data_split_y[0],axis=0)','skew(data_split_y[1],axis=0)',
        'kurtosis(data_z,axis=0)','kurtosis(data_split_x[4],axis=0)','data_split_y[2].min(axis=0)',
        'kurtosis(data_split_z[2],axis=0)','skew(data_split_y[4],axis=0)','data_split_y[1].var(axis=0)',
        'data_split_x[3].var(axis=0)','kurtosis(data_split_z[0],axis=0)','data_split_y[3].min(axis=0)',
        'kurtosis(data_split_z[3],axis=0)','data_split_y[0].std(axis=0)','skew(data_split_x[4],axis=0)',
        'skew(data_split_x[0],axis=0)','skew(data_split_z[4],axis=0)','data_y.min(axis=0)']
    else: # For the most representative component (X)


        data = pd.read_csv(files, sep=' ', header=None, names=['x']) 
        
        #Conversion from 0-63 to m/s^2
        df_x = -14.709 + (data.iloc[:,0:1]/63)*(2*14.709)
        
        """
        #Median filtering
        x = np.median(strided_app(df_x.values.flatten(), 3,1),axis=1)

        df_x = pd.DataFrame(x, columns=['x'])
        """
        data_x = df_x.values

        #Divide data in segments
        split_index=5 #Number of segments
        data_split_x=np.array_split(data_x, split_index)

        appended_before=['data_split_x[2].min(axis=0)','data_x.min(axis=0)','data_split_x[2].mean(axis=0)',
        'np.median(data_split_x[2],axis=0)','data_split_x[1].min(axis=0)','data_split_x[1].mean(axis=0)',
        'data_x.mean(axis=0)','np.median(data_split_x[1],axis=0)','np.median(data_x,axis=0)',
        'data_split_x[3].mean(axis=0)','np.median(data_split_x[3],axis=0)','data_split_x[3].min(axis=0)',
        'data_split_x[4].min(axis=0)','data_split_x[2].max(axis=0)','data_split_x[2].std(axis=0)',
        'data_split_x[4].mean(axis=0)','data_split_x[3].max(axis=0)','np.median(data_split_x[4],axis=0)',
        'mad(data_split_x[2],axis=0)','data_split_x[0].min(axis=0)','np.median(data_split_x[0],axis=0)',
        'data_split_x[0].mean(axis=0)','data_split_x[1].max(axis=0)','data_split_x[4].var(axis=0)',
        'data_split_x[2].var(axis=0)','data_split_x[4].std(axis=0)','mad(data_split_x[4],axis=0)',
        'data_split_x[0].var(axis=0)','data_x.var(axis=0)','data_split_x[0].std(axis=0)',
        'data_split_x[4].max(axis=0)','data_x.std(axis=0)','mad(data_split_x[0],axis=0)',
        'data_split_x[1].std(axis=0)','data_x.max(axis=0)','data_split_x[1].var(axis=0)',
        'data_split_x[0].max(axis=0)','mad(data_x,axis=0)','kurtosis(data_x,axis=0)',
        'mad(data_split_x[1],axis=0)','skew(data_split_x[3],axis=0)','mad(data_split_x[3],axis=0)',
        'skew(data_split_x[1],axis=0)','kurtosis(data_split_x[2],axis=0)','skew(data_split_x[2],axis=0)',
        'skew(data_x,axis=0)','kurtosis(data_split_x[0],axis=0)','kurtosis(data_split_x[1],axis=0)',
        'kurtosis(data_split_x[3],axis=0)','data_split_x[3].std(axis=0)','kurtosis(data_split_x[4],axis=0)',
        'data_split_x[3].var(axis=0)','skew(data_split_x[4],axis=0)','skew(data_split_x[0],axis=0)']

    #Create initial_features_matrix
    appended_features_split=[]
    appended_features=[]

    for i in range (0, features):
        appended_features_before = eval(appended_before[features])
        appended_features.append(appended_features_before[0])
        appended_features_before=[] 

    appended_features_all.append(appended_features)

    appended_features_df = pd.DataFrame(appended_features_all)
 
    #Binarize detectiom
    if wrist_class[cls] != 1:
        wrist_class[cls] = 0 #Other classes than drink are considered as CLASS 0. Drink = CLASS 1
    
    #Access to dictionary class number and add it as a feature
    appended_features_df[-1]= wrist_class[cls]
    
    #Return table containing all rows for every class and feature colums 
    #(mean*3, sd*3, Max*3, Min*3, Y). Number of the row is manteined. (0~101)
    return appended_features_df




time_avg_array=[]
time_std_array=[]
n=0

if __name__ == "__main__":
    for i in range (0,best_features_number):
        print ("Number of features: "+str(i+1))
        wrist_df = gather(wrist_class,features=i+1)

        #Extact number of the class (feature Y)
        wrist_Y = np.asarray(wrist_df.iloc[:, -1])

        #Total number of features
        num_features=len(wrist_df.columns)-1

        #Extact rest of colums (features)
        wrist_X = np.asarray(wrist_df.iloc[:,:num_features])
        
        #Normalize data (0-1)
        min_max_scaler = preprocessing.MinMaxScaler()
        wrist_X= min_max_scaler.fit_transform(wrist_df.iloc[:,:num_features])
        wrist_X_df = pd.DataFrame(wrist_X, columns=(wrist_df.iloc[:,:num_features]).columns)

        #Create model for every algorithm
        wrist_lg = LogisticRegression()
        wrist_lg_model = wrist_lg.fit(wrist_X, wrist_Y)

        wrist_rf = RandomForestClassifier(verbose= 0,n_estimators= 100,random_state= n)
        wrist_rf_model = wrist_rf.fit(wrist_X, wrist_Y)

        wrist_knn = KNeighborsClassifier(n_neighbors=3)
        wrist_knn_model = wrist_knn.fit(wrist_X, wrist_Y)

        wrist_nb = GaussianNB()
        wrist_nb_model = wrist_nb.fit(wrist_X, wrist_Y)

        wrist_svm = svm.SVC(kernel='linear', C=64.0, probability=True,random_state = n)
        wrist_svm_model = wrist_svm.fit(wrist_X, wrist_Y)

        wrist_mlp = MLPClassifier(hidden_layer_sizes=(16, 16, 16), max_iter=1000,random_state = n)
        wrist_mlp_model= wrist_mlp.fit(wrist_X, wrist_Y)  

        wrist_dt = tree.DecisionTreeClassifier(random_state=n)
        wrist_dt_model = wrist_dt.fit(wrist_X, wrist_Y)


        #Measure data processing times for inference
        wrist_df_test = gather_one(wrist_class, features=i+1)
        timer = get_ipython().run_cell_magic('timeit', '-n 3 -r 10 -o', '\nwrist_df_test = gather_one(wrist_class,features=i+1)\nwrist_Y_test = np.asarray(wrist_df_test.iloc[:-1])\nnum_features=len(wrist_df.columns)-1\nwrist_X_test = np.asarray(wrist_df_test.iloc[:,:num_features])\nmin_max_scaler = preprocessing.MinMaxScaler()\nwrist_X_test= min_max_scaler.fit_transform(wrist_df_test.iloc[:,:num_features])\nwrist_X_df_test = pd.DataFrame(wrist_X, columns=(wrist_df_test.iloc[:,:num_features]).columns)')

        #Obtain average and standard deviation values
        timer_avg = (np.mean(timer.timings))
        timer_std = (np.std(timer.timings))

        #Save values array
        time_avg_array.append(timer_avg)
        time_std_array.append(timer_std)

        print ("Mean "+ str(timer_avg) + " -- Std "+ str(timer_std))


    #Save data in .txt file
    get_ipython().run_cell_magic('capture', 'cap --no-stderr', 'print (time_avg_array)\nprint(time_std_array)')
    with open('Multiprocessing_Inference_processing_times_'+str(number_components)+'Comp_'+str(best_features_number)+'Features.txt', 'w') as f:
        f.write(cap.stdout)      