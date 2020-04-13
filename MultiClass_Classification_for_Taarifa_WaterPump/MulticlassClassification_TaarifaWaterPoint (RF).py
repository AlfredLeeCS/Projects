#!/usr/bin/env python
# coding: utf-8

# ### Project Description : This is competition hosted by Data Driven. The objective is to predict which water pumps are functional, which need some repairs, and which don't work at all, a multiclass classifcation problem. The model produced managed to get 81.06% accuracy on test data, currently on top 14% of the leaderboeard.

# ##### Hypothesis / Assumptions made as following:-
# ##### 1. 'Date Recorded' which is described as the date row of observations is recorded will be assumed to be the date waterpoint is installed.
# ##### 2. Missing value for the field 'Construction Year' is assumed to be omitted unintentionally and will be imputed with the mean/other better method.
# ##### 3. Numerical features such as  'Amount_tsh', 'Population', 'GPS Height' & 'Longitude' are having value 0 and they will be assumed to be invalid and will be imputed. 
# ##### 4. Categorical features such as 'permit', ‘scheme_management‘ & ‘public_meeting are columns with valid value and missing value is assumed to be omitted unintentionally and will be imputed.
# 

# ### 1. Import Dependencies Library

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import xgboost as xgb

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from boruta import BorutaPy
from fancyimpute import IterativeImputer as MICE
from datetime import datetime

import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots


# ### 2. Load train & test set data

# In[ ]:


df_train = pd.read_csv("drive/My Drive/Data_Interchange/train_taarifa.csv",parse_dates=True)
y_train = pd.read_csv("drive/My Drive/Data_Interchange/y_train.csv")
df_test = pd.read_csv("drive/My Drive/Data_Interchange/test_taarifa.csv",parse_dates=True)
df_full = pd.concat([df_train,df_test],sort=False).drop('status_group',axis=1)

df_train_index = df_train.index.values
df_test_index = df_test.index.values

print(df_full.shape)
print(df_train.shape)
print(y_train.shape)
print(df_test.shape)


# ### 3. Exploratory Data Analysis

# In[ ]:


df_train.dtypes


# In[ ]:


# Check if there is any missing data
# There are 7 columns with missing data
df_train.isnull().sum()


# In[ ]:


# Check the counts of target variable

df_train['status_group'].value_counts()


# In[ ]:


# Plot the bar chart to visualize target "status_group"

plt.figure(figsize=(5, 5), dpi= 80, facecolor='w', edgecolor='k')

palette=[sns.color_palette()[3],sns.color_palette()[6],sns.color_palette()[4]]
y_train.status_group.value_counts().plot(kind='barh', color=palette)


# In[ ]:


# Try visualize target "status_group" & plot its relationship with "construction_year"
# Relate to hypothesis 1

df_train.construction_year=pd.to_numeric(df_train.construction_year)
df_train.loc[df_train.construction_year <= 0, df_train.columns=='construction_year'] = 1950

hist1=df_train[df_train.status_group == 'functional'].construction_year
hist2=df_train[df_train.status_group == 'functional needs repair'].construction_year
hist3=df_train[df_train.status_group == 'non functional'].construction_year

plt.figure(figsize=(6, 9), dpi= 80, facecolor='w', edgecolor='k')
n,b,p=plt.hist([hist1, hist2, hist3], stacked=True,range=[1950,2010])
plt.legend(['functional','functional needs repair','non functional'],loc=0)
plt.text(1952, 15000,'N/A',fontsize=20,rotation=90,color='white')
plt.xlabel('Construction Year', fontsize=18)


# In[ ]:


#convert status group label into numerical data as training take numerical inputs

val_status_group={'functional':2, 'functional needs repair':1,
                   'non functional':0}
df_train['status_group_vals']=df_train.status_group.replace(val_status_group)


# ### 3. Data Transformation - Part 1

# ##### Treatment for categorical data

# In[ ]:


# For all categorical data, checking out the number of category in the field, only kept ideally the top 5, else as others.
df_train['funder'].value_counts()


# In[ ]:


# Divide all categorical columns into max 5 categories, else as others
# This will helps in model convergence
def top5bin_funder(row):
    if row['funder']=='Government Of Tanzania':
        return 'government'
    elif row['funder']=='Danida':
        return 'danida'
    elif row['funder']=='Hesawa':
        return 'hesawa'
    elif row['funder']=='Rwssp':
        return 'rwssp'
    elif row['funder']=='World Bank':
        return 'world_bank'    
    else:
        return 'other'

def top5bin_cleaning(row):
    if row['installer']=='DWE':
        return 'dwe'
    elif row['installer']=='Government':
        return 'government'
    elif row['installer']=='RWE':
        return 'rwe'
    elif row['installer']=='Commu':
        return 'commu'
    elif row['installer']=='DANIDA':
        return 'danida'    
    else:
        return 'other'

def top5bin_scheme(row):
    '''Keep top 5 values and set the rest to 'other'. '''
    if row['scheme_management']=='VWC':
        return 'vwc'
    elif row['scheme_management']=='WUG':
        return 'wug'
    elif row['scheme_management']=='Water authority':
        return 'wtr_auth'
    elif row['scheme_management']=='WUA':
        return 'wua'
    elif row['scheme_management']=='Water Board':
        return 'wtr_brd'
    else:
        return 'other'

df_train['funder']= df_train.apply(lambda row: top5bin_funder(row), axis=1)
df_test['funder']= df_test.apply(lambda row: top5bin_funder(row), axis=1)
df_train['installer']= df_train.apply(lambda row: top5bin_cleaning(row), axis=1)
df_test['installer']= df_test.apply(lambda row: top5bin_cleaning(row), axis=1)
df_train['scheme_management'] = df_train.apply(lambda row: top5bin_scheme(row), axis=1)
df_test['scheme_management'] = df_test.apply(lambda row: top5bin_scheme(row), axis=1)


# ##### Drop all the columns which are redundant columns with too many unique values, this would not bring any goods to model

# In[ ]:


df_train['waterpoint_type'].value_counts()


# In[ ]:


df_train['waterpoint_type_group'].value_counts()


# In[ ]:


# Eg. A few pairs of variables are similar :
# (waterpoint_type,waterpoint_type_group) & (source & source type) & etcs both are almost similar, can drop one of them

df_train=df_train.drop(['subvillage','scheme_name','recorded_by','waterpoint_type','source','quantity',
                        'payment','management','extraction_type','extraction_type_group','quality_group'],axis=1)
df_test=df_test.drop(['subvillage','scheme_name','recorded_by','waterpoint_type','source','quantity',
                      'payment','management','extraction_type','extraction_type_group','quality_group'],axis=1)


# ##### Fill the valid columns which contain blank value with "Unknown"/"Other"

# In[ ]:


df_train['scheme_management'].value_counts()


# In[ ]:


# Value : true and false, others imputed as unknown.
df_train.public_meeting = df_train.public_meeting.fillna('Unknown')
df_test.public_meeting = df_test.public_meeting.fillna('Unknown')
df_train.permit = df_train.permit.fillna('Unknown')
df_test.permit = df_test.permit.fillna('Unknown')

df_train.scheme_management = df_train.scheme_management.fillna('other')
df_test.scheme_management = df_test.scheme_management.fillna('other')


# In[ ]:


# Checking Null Values on train data, all data has been imputed accordingly
df_train.apply(lambda x: sum(x.isnull()), axis=0)


# In[ ]:


# Statistics for full train/test dataset
df_full.describe()


# In[ ]:


# Visualize of categorical column after transformation against target Y

plt.figure(figsize=(24,32))

# columns to plot
df_cat_cols=['quantity_group','water_quality','waterpoint_type_group','extraction_type_class','scheme_management',
             'permit','funder','installer','source_type','source_class','payment_type','management_group',
             'public_meeting','basin'] 

df_col_i = df_train.loc[:, df_cat_cols]
df_i = pd.concat([df_col_i, df_train['status_group']], axis=1)

for i, col in enumerate(df_cat_cols):
    plt.subplot(4, 4, i+1)
    sns.countplot(df_i[col], hue='status_group', data=df_i)
    plt.title(str(col))
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    plt.xticks(rotation=90)
plt.savefig('cat_col_with_label.png',dpi=300)
plt.show()


# ### Data Imputation (Mean/Median - depend on the distribution)
# ##### Tried IterativeImputer but doesn't work out so well so stick with mean/median.
# #####  *** These are variables with invalid 0 value and need to be treated seperately.

# In[ ]:


### Imputation for invalid value 0 in numerical column with mean/median

# amount_tsh median equal to 0, no replacement done in first consideration 

def population_impute(row):
    if row['population']==0:
        return 25 #replace with median, skewed data
    else:
        return row['population']

def gpsheight_impute(row):
    if row['gps_height']==0:
        return 364 #replace with median, skewed data
    else:
        return row['gps_height']

def longitude_impute(row):
    if row['longitude']==0:
        return 34.074262
    else:
        return row['longitude']

df_train['population']= df_train.apply(lambda row: population_impute(row), axis=1)
df_test['population']= df_test.apply(lambda row: population_impute(row), axis=1)
df_train['gps_height']= df_train.apply(lambda row: gpsheight_impute(row), axis=1)
df_test['gps_height']= df_test.apply(lambda row: gpsheight_impute(row), axis=1)
df_train['longitude']= df_train.apply(lambda row: longitude_impute(row), axis=1)
df_test['longitude']= df_test.apply(lambda row: longitude_impute(row), axis=1)


# ### Feature engineering: Construction tenure can be useful to know the period of pump installed

# In[ ]:


df_train['date_recorded'] = pd.to_datetime(df_train['date_recorded'])
df_train['recorded_year'] = df_train['date_recorded'].dt.year
df_test['date_recorded'] = pd.to_datetime(df_test['date_recorded'])
df_test['recorded_year'] = df_test['date_recorded'].dt.year

def construct_tenure(row):
    if row['construction_year'] !=0:
        return (row['recorded_year']- row['construction_year'])
    else:
        return 15.295624 #replace with mean
df_train['construct_tenure'] = df_train.apply(lambda row: construct_tenure(row), axis=1)
df_test['construct_tenure'] = df_test.apply(lambda row: construct_tenure(row), axis=1)

df_train = df_train.drop(["construction_year","recorded_year","date_recorded"],axis=1)
df_test = df_test.drop(["construction_year","recorded_year","date_recorded"],axis=1)


# In[ ]:


# Data is clean for model training
# not too correlated to each other, good fit for model
df_train.corr()


# In[ ]:


# Lastly drop those variable that seem superflous like 'num_private' & 'wpt_name which no extra info

df_train = df_train.drop(['num_private','wpt_name','lga','region','ward','status_group','id'], axis=1)
df_test = df_test.drop(['num_private','wpt_name','lga','region','ward','id'], axis=1)


# ##### One-hot encoding for categorical data

# In[ ]:


# Get dummy columns for the categorical columns and shuffle the data.
## As most model only can take numerical data as inputs.

dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management', 'permit',
              'extraction_type_class','management_group', 'payment_type', 'water_quality',
              'quantity_group', 'source_type', 'source_class','waterpoint_type_group'] #construction_year

df_train = pd.get_dummies(df_train, columns = dummy_cols)
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_test = pd.get_dummies(df_test, columns = dummy_cols)


# In[ ]:


print('Shape of training data',df_train.shape)
print('Shape of testing data',df_test.shape)


# In[ ]:


df_train.corr()


# In[ ]:


# From the above correaltion table, it seem taht 
# waterpoint_type_group_hand pump is highly correlated with extraction_type_class_handpump,
# so will proceed to drop one of them

df_train=df_train.drop(['waterpoint_type_group_hand pump'],axis=1)
df_test=df_test.drop(['waterpoint_type_group_hand pump'],axis=1)

df_train=df_train.drop(['source_type_other'],axis=1)
df_test=df_test.drop(['source_type_other'],axis=1)


# In[ ]:


# Define Y & drop it from the train set to proceed for feature selection 

df_train_cp = df_train.copy()
target = df_train.status_group_vals
features = df_train.drop('status_group_vals', axis=1)
features1=features.copy()


# ### 4. Feature Selection / Feature Reduction 

# In[ ]:


# Set up timer to time feature selection process

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[ ]:


# RandomForrestClassifier is used here as the estimator for Boruta. 
# The max_depth of the tree is advised to be between 3 to 7 for better result, set most setting to default.

rf = RandomForestClassifier(criterion='gini',
                                n_estimators=500,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1,
                                max_depth=6)

X_boruta=features.values
y_boruta=target.values

boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2)
start_time = timer(None)
boruta_selector.fit(X_boruta,y_boruta)
timer(start_time)


# In[ ]:


# number of selected features after feature selection process

print ('\n Number of selected features:')
print (boruta_selector.n_features_)


# In[ ]:


# Put selected list into pandas DataFrame in ascending sort

features1=pd.DataFrame(features.columns.tolist())
features1['rank']=boruta_selector.ranking_
features1 = features1.sort_values('rank', ascending=True).reset_index(drop=True)
print ('\n Top %d features:' % boruta_selector.n_features_)
print (features1.head(boruta_selector.n_features_))


# In[ ]:


# Boruta output with full list of features with ranking sorted ascendingly

features1=pd.DataFrame(features.columns.tolist())
features1['rank']=boruta_selector.ranking_
features1 = features1.sort_values('rank', ascending=True).reset_index(drop=True)
features1


# In[ ]:


#Drop columns which Boruta didnt give/predict rank 1 importance

df_train=features.drop(['water_quality_milky','source_type_dam','payment_type_on failure','funder_hesawa',
                        'management_group_parastatal','water_quality_salty abandoned','management_group_unknown',
                        'management_group_other','extraction_type_class_rope pump','payment_type_other',
                        'installer_danida','water_quality_fluoride','source_class_unknown','water_quality_coloured',
                        'extraction_type_class_wind-powered','waterpoint_type_group_cattle trough',
                        'waterpoint_type_group_dam','water_quality_fluoride abandoned'],axis=1)

df_test=df_test.drop(['water_quality_milky','source_type_dam','payment_type_on failure','funder_hesawa',
                      'management_group_parastatal','water_quality_salty abandoned','management_group_unknown',
                      'management_group_other','extraction_type_class_rope pump','payment_type_other','installer_danida',
                      'water_quality_fluoride','source_class_unknown','water_quality_coloured','extraction_type_class_wind-powered',
                      'waterpoint_type_group_cattle trough','waterpoint_type_group_dam','water_quality_fluoride abandoned'],axis=1)


# In[ ]:


# Print final train & test set shape
print(df_train.shape)
print(df_test.shape)


# In[ ]:


# # Invalid value like 0 in some columns detected, will be imputed with IterativeImputer later

# def population_cleaning(train):
#     if train['population']==0:
#         return np.nan
#     else:
#         return train['population']

# def gpsheight_cleaning(train):
#     if train['gps_height']==0:
#         return np.nan
#     else:
#         return train['gps_height']

# def longitude_cleaning(train):
#     if train['longitude']==0:
#         return np.nan
#     else:
#         return train['longitude']

# def amounttsh_cleaning(train):
#     if train['amount_tsh']==0:
#         return np.nan
#     else:
#         return train['amount_tsh']

# df_train['population']= df_train.apply(lambda row: population_cleaning(row), axis=1)
# df_test['population']= df_test.apply(lambda row: population_cleaning(row), axis=1)
# df_train['gps_height']= df_train.apply(lambda row: gpsheight_cleaning(row), axis=1)
# df_test['gps_height']= df_test.apply(lambda row: gpsheight_cleaning(row), axis=1)
# df_train['longitude']= df_train.apply(lambda row: longitude_cleaning(row), axis=1)
# df_test['longitude']= df_test.apply(lambda row: longitude_cleaning(row), axis=1)
# df_train['amount_tsh']= df_train.apply(lambda row: amounttsh_cleaning(row), axis=1)
# df_test['amount_tsh']= df_test.apply(lambda row: amounttsh_cleaning(row), axis=1)


# In[ ]:


# # Check and see if there is missing value now as those invalid value 0 has been replaced with np.nan
# df_train.isnull().sum()


# In[ ]:


# Save a copy for data normalization and scaler
df_train1 = df_train.copy()
df_test1 = df_test.copy()


# In[ ]:


# Get numerical & categorical columns
df_train_num_col = df_train1.select_dtypes(exclude=['category','object']).columns
df_train_cat_col = df_train1.select_dtypes(include=['category','object']).columns

print("Numeric cols : ", len(df_train_num_col))
print("Numeric cols : ", len(df_train_cat_col))


# In[ ]:


# Prepare for data transformation - Part 2 : Numerical data
# Visualize the relationship / distribution of numerical data with target - Boxplot

plt.figure(figsize=(8,18))

# columns to plot
df_num_cols=['amount_tsh','longitude','region_code', 'latitude','district_code','population', 'gps_height','construct_tenure'] 

df_col_i = df_train1.loc[:, df_num_cols]
df_i = pd.concat([df_col_i, df_train_cp['status_group_vals']], axis=1)

for i, col in enumerate(df_num_cols):
    plt.subplot(4, 2, i+1)
    sns.boxenplot(y=df_i[col], x=df_i['status_group_vals'])
    plt.title(str(col))
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    # plt.yscale('log')
plt.savefig('num_col_with_label.png',dpi=300)
plt.show()


# In[ ]:


# Scatterplot with target Y

plt.figure(figsize=(24, 32), dpi= 80, facecolor='w', edgecolor='k')

df_col_i = df_train1.loc[:, df_num_cols]
df_i = pd.concat([df_col_i, df_train_cp['status_group_vals']], axis=1)

sns.pairplot(df_i, hue='status_group_vals')
plt.yscale('log')
plt.show()


# ### 5. Split Data into train & validation set
# 
# ##### Chosen Train 90%: Validation 10% split as it turn out to give best accuracy.

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(df_train1, target, train_size=0.9, random_state=1085)


# ##### Perform Data Transformation - Part 2 (after train_test_split)

# ### Tried IterativeImputer (Imputation with multiple regression) but imputation only manage to give validation accuracy of 80.4% hence will not proceed for this case.

# In[ ]:


# # FancyImpute with MICE
# # Default method used is BayesianRidge

# # Use Logistic regression as the baseline
# logreg = make_pipeline(RobustScaler(), LogisticRegression())

# # Impute invalid/missing value
# # Fit_transform on training set
# mice = MICE(verbose=0)
# X_train_fancy_mice = mice.fit_transform(X_train)
# scores = cross_val_score(logreg, X_train_fancy_mice, y_train, cv=10)
# scores.mean()


# In[ ]:


# # Then apply transformation to validation and test dataset to avoid data leakage
# col_list = X_train.columns.tolist()
# X_train_mice = pd.DataFrame(X_train_fancy_mice, columns=col_list)
# print(X_train_mice.shape)

# X_train_mice.head()


# In[ ]:


# # Check and see if there is missing value after MICE imputation
# # The invalid value has been imputed
# X_train_mice.isnull().sum()


# In[ ]:


# # Apply transformation on validation & test dataset

# X_val_fancy_mice = mice.transform(X_val)
# df_test_fancy_mice = mice.transform(df_test)

# X_val_mice = pd.DataFrame(X_val_fancy_mice, columns=col_list)
# df_test_mice = pd.DataFrame(df_test_fancy_mice, columns=col_list)
# print(X_val_mice.shape)
# print(df_test_mice.shape)


# ### Data Normalization / Scaling 
# ##### Good in removing outliers and provide better data distribution for model convergence.

# In[ ]:


# Columns intended for normalization/scaling
df_num_cols=['amount_tsh','longitude','region_code', 'latitude','district_code','population','gps_height','construct_tenure'] 


# In[ ]:


# Plot the distribution of numerical columns with histogram
plt.figure(figsize=(12,8))

for i, col in enumerate(df_num_cols):
    plt.subplot(4, 2, i + 1)
    sns.distplot(X_train[col])
    plt.title(str(col))
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    plt.tight_layout()
plt.show()


# In[ ]:


# Do normalization and scaling

# Train set normalization
X_train.amount_tsh = X_train.amount_tsh.apply(lambda x: np.log10(x+1))
X_train.longitude = X_train.longitude.apply(lambda x: np.log10(x+1))
X_train.region_code = X_train.region_code.apply(lambda x: np.log10(x+1))
X_train.district_code = X_train.district_code.apply(lambda x: np.log10(x+1))
X_train.population = X_train.population.apply(lambda x: np.log10(x+1))

# Validation set normalization
X_val.amount_tsh = X_val.amount_tsh.apply(lambda x: np.log10(x+1))
X_val.longitude = X_val.longitude.apply(lambda x: np.log10(x+1))
X_val.region_code = X_val.region_code.apply(lambda x: np.log10(x+1))
X_val.district_code = X_val.district_code.apply(lambda x: np.log10(x+1))
X_val.population = X_val.population.apply(lambda x: np.log10(x+1))

# Test set normalization
df_test1.amount_tsh = df_test1.amount_tsh.apply(lambda x: np.log10(x+1))
df_test1.longitude = df_test1.longitude.apply(lambda x: np.log10(x+1))
df_test1.region_code = df_test1.region_code.apply(lambda x: np.log10(x+1))
df_test1.district_code = df_test1.district_code.apply(lambda x: np.log10(x+1))
df_test1.population = df_test1.population.apply(lambda x: np.log10(x+1))


# In[ ]:


# # df_train1=df_train1.fillna(df_train1.mean())
# df_test1=df_test1.fillna(df_test1.mean())
# X_train_mice=X_train_mice.fillna(X_train_mice.mean())
# X_val_mice=X_val_mice.fillna(X_val_mice.mean())


# In[ ]:


# Data distribution after normalization
plt.figure(figsize=(12,8))

for i, col in enumerate(df_num_cols):
    plt.subplot(4, 2, i + 1)
    sns.distplot(X_train[col])
    plt.title(str(col))
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    plt.tight_layout()
plt.show()


# ##### Data Scaling : To make data robust to outliers

# In[ ]:


# Scale the numerical variables can help model find pattern and faster convergence
# Gradient Boosting/Ensembles Tree algorithms is still prone to unnormalized/unscaled data due to bagging & boosting
rs =RobustScaler()

# Fit_transform should be done on only training data to avoid data leakage problem
X_train[df_num_cols] = rs.fit_transform(X_train[df_num_cols])

X_val[df_num_cols] = rs.transform(X_val[df_num_cols])
df_test1[df_num_cols] = rs.transform(df_test1[df_num_cols])


# ##### Data are now good to proceed to model training ...

# ### 6. Model Training

# ##### Hyperparameter tuning with GridSearchCV

# In[ ]:


# Model 1 : Linear Kernel SVM Classifier

def SVM_model(X_train, X_val, y_train, y_val):
    if __name__ == '__main__':
        
        #scl = StandardScaler()
        svm = LinearSVC()
        
        parameters = {'C':[0.001,0.01,0.1,1.0,10.0,100.0],'class_weight':[None, 'balanced']}

        estimator = GridSearchCV(svm, parameters,n_jobs=-1)

        estimator.fit(X_train, y_train)

        best_params = estimator.best_params_
                                 
        validation_accuracy = estimator.score(X_val, y_val)
        print('Validation accuracy: ', validation_accuracy)
        print(best_params)


# In[ ]:


SVM_model(X_train, X_val, y_train, y_val)


# In[ ]:


# Model 2 : GBM

def GradientBoostingClassifier_model(X_train, X_val, y_train, y_val):
    if __name__ == '__main__':
        
        
        gb = GradientBoostingClassifier()
        
        parameters = {'learning_rate': [0.1],
                     'max_depth': [10],
                     'min_samples_leaf': [16],
                     "min_samples_split" : [8,12,16],
                    #  'max_features': ["log2","sqrt"],
                     "subsample":[0.5, 0.8, 0.95],
                     'n_estimators': [200]}
        
        estimator = GridSearchCV(gb,n_jobs=-1,param_grid=parameters)
        
        estimator.fit(X_train, y_train)

        best_params = estimator.best_params_
                                 
        validation_accuracy = estimator.score(X_val, y_val)
        print('Validation accuracy: ', validation_accuracy)
        print(best_params)


# In[ ]:


GradientBoostingClassifier_model(X_train, X_val, y_train, y_val)


# In[ ]:


# Model 3 : Random Forest

def RandomForestClassifier_model(X_train, X_val, y_train, y_val):
    if __name__ == '__main__':
        
        
        rf = RandomForestClassifier()
        
        parameters = {'max_samples' : [0.8],
                      'max_depth': [16,22],
                      'n_estimators': [500,1000]}
        
        estimator = GridSearchCV(rf,n_jobs=-1,param_grid=parameters,verbose=2,cv=5)

        estimator.fit(X_train, y_train)

        best_params = estimator.best_params_
                                 
        validation_accuracy = estimator.score(X_val, y_val)
        print('Validation accuracy: ', validation_accuracy)
        print(best_params)


# In[ ]:


RandomForestClassifier_model(X_train, X_val, y_train, y_val) 


# In[ ]:


rf = RandomForestClassifier(
                      max_samples = 0.8,
                      max_depth= 22,
                    #   n_jobs=1,
                    #   oob_score=True,
                      n_estimators= 1000)
 

rf.fit(X_train, y_train)


# In[ ]:


rf.score(X_val, y_val)


# In[ ]:


rf_predictions = rf.predict(df_test1)

test_rf = pd.DataFrame(rf_predictions,columns=['status_group'])

predict_rf1 = pd.concat([test_rf, df_test], axis=1)

predict_rf1a = predict_rf1[['id','status_group']]
predict_rf1a = predict_rf1a.astype({"id":'object', "status_group":'object'}) 

def parse_values(x):
    if x == 2: 
        return 'functional'
    elif x == 1: 
        return 'functional needs repair'
    elif x == 0: 
        return 'non functional'

predict_rf1a['status_group'] = predict_rf1a['status_group'].apply(parse_values)

predict_rf1a = predict_rf1a.reset_index(drop=True).set_index('id')

predict_rf1a.to_csv('pred_rf.csv')


# In[ ]:


# Model 4 : XGBoost

def XGBoost_model(X_train, X_val, y_train, y_val):
    if __name__ == '__main__':
        
        
        xgboost = xgb.XGBClassifier()
        
        parameters = {'nthread':[4], 
                      'objective':['multi:softprob'],
                      'num_class':[3],
                      'eta': [0.05],
                      'max_depth': [10,15],
                    #   'lambda' : [0.01],
                      'alpha' : [0.1],
                      'colsample_bylevel': [0.6],
                      'colsample_bytree': [0.7],
                      'seed': [103734],
                      'n_estimators': [100,200]}
        
        estimator = GridSearchCV(xgboost,n_jobs=-1,param_grid=parameters,
                                 cv = 5,
                                 scoring='accuracy',
                                 verbose=3,refit=True)
        

        estimator.fit(X_train, y_train)

        best_params = estimator.best_params_
                                 
        validation_accuracy = estimator.score(X_val, y_val)
        print('Validation accuracy: ', validation_accuracy)
        print(best_params)


# In[ ]:


XGBoost_model(X_train, X_val, y_train, y_val)


# In[ ]:


#final for xgboost

xgbf = xgb.XGBClassifier(nthread=4, 
                      objective='multi:softprob',
                      num_class=3,
                      eta=0.05,
                      max_depth=15,
                      max_bin=256,
                      min_child_weight=0.1,
                      gamma=0.0,
                    #   lambda = 0.01,
                      alpha =0.1,
                      colsample_bylevel=0.6,
                      colsample_bytree=0.7,
                      seed=103734,
                      n_estimators=100)
        

xgbf.fit(X_train, y_train)


# In[ ]:


xgbf.score(X_val,y_val)


# ### Final Model for submission : Random Forest (Validation Accuracy : 0.8181)
# 
# 

# In[ ]:


# RF Model

rf = RandomForestClassifier(
                      max_samples = 0.8,
                      max_depth= 22,
                    #   n_jobs=1,
                    #   oob_score=True,
                      n_estimators= 1000)
 

rf.fit(X_train, y_train)

# Inference
rf_predictions = rf.predict(df_test1)

test_rf = pd.DataFrame(rf_predictions,columns=['status_group'])

predict_rf1 = pd.concat([test_rf, df_test], axis=1)

predict_rf1a = predict_rf1[['id','status_group']]
predict_rf1a = predict_rf1a.astype({"id":'object', "status_group":'object'}) 

def parse_values(x):
    if x == 2: 
        return 'functional'
    elif x == 1: 
        return 'functional needs repair'
    elif x == 0: 
        return 'non functional'

predict_rf1a['status_group'] = predict_rf1a['status_group'].apply(parse_values)

predict_rf1a = predict_rf1a.reset_index(drop=True).set_index('id')

predict_rf1a.to_csv('pred_rf.csv')


# ### 7. Model Visualization : Features Importance, Permutation Importance & Partial Dependence Plot
# 
# 

# In[ ]:


# Plotting features importance

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

features = df_train1.columns.tolist()
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### Permutation Importance
# 
# ##### The first number in each row shows how much model performance decreased with a random shuffling (in this case, using "accuracy" as the performance metric).
# 
# ##### The permutation importance is the measure of performance of the respective column with randomness introduced in the column by repeating the process with multiple shuffles. The number after the ± measures how performance varied from one-reshuffling to the next.

# In[ ]:


### Permutation Importance

perm = PermutationImportance(rf, random_state=1).fit(X_val, y_val)
eli5.show_weights(perm, feature_names = X_val.columns.tolist())


# ### Partial Dependence Plot
# ##### Partial dependence plot (PDP) is good at visualizing & capturing realistic patterns in the model. Y axis of the plot is interpreted as change in the prediction from what it would be predicted at the baseline or leftmost value while blue shaded area indicates level of confidence.
# --------------------------------------------------------------------------
# ##### Below are some analysis findings:-
# ##### 1) When the water point is dry, it tends to increase the non-functionality chances of the water point.
# ##### 2) As the construction tenure increase up to 30 years, it tends to increase the malfunctioned chances of the water point. However after 30 years, the chances of non-functionality will remain the same.
# ##### 3) When the water point group belongs to others (instead of communal standpipe and etc.), it increases the chances of non-functionality.
# ##### 4) When the extraction type class belongs to others (instead of gravity, hand pump and etc.), it increases the chances of non-functionality.
# ##### 5) The chances of non-functionality peak when the population size is at its median and there is little marginal effect of population size thereafter.
# 

# In[ ]:


# Plot Partial Dependence Plot for top permutation important & feature important variables
# PDP 1a - Quantity_Group_Dry

feature_names = df_train.columns.tolist()
feature_to_plot = 'quantity_group_dry'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 1b - Quantity_Group_Enough

feature_names = df_train.columns.tolist()
feature_to_plot = 'quantity_group_enough'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 2 - Construction Tenure 
feature_names = df_train.columns.tolist()
feature_to_plot = 'construct_tenure'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 3a - Waterpoint Group
feature_names = df_train.columns.tolist()
feature_to_plot = 'waterpoint_type_group_other'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 3b - Waterpoint Group
feature_names = df_train.columns.tolist() 
feature_to_plot = 'waterpoint_type_group_communal standpipe' ## Majority 58% but no significant impact on the model

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 4 - Longitude
feature_names = df_train.columns.tolist()
feature_to_plot = 'longitude'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 5 - Latitude
feature_names = df_train.columns.tolist()
feature_to_plot = 'latitude'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 6 - Extraction Type Class 
feature_names = df_train.columns.tolist()
feature_to_plot = 'extraction_type_class_other'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 7 - Population 
feature_names = df_train.columns.tolist()
feature_to_plot = 'population'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 8 - GPS height 
feature_names = df_train.columns.tolist()
feature_to_plot = 'gps_height'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# PDP 9 - Scheme Management
feature_names = df_train.columns.tolist()
feature_to_plot = 'scheme_management_vwc'

pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_val, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


# Test dataset get accuracy of 81.06%, currently top 14% in the leaderboard.

