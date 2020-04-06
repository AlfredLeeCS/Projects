# -*- coding: utf-8 -*-
"""MultiClass_Classification_for_Taarifa_WaterPump (RF).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13KksVMfjc29209o1pKXzxOMGVOp0U9Qb

### Project Description : This is competition hosted by Data Driven. The objective is to predict which water pumps are functional, which need some repairs, and which don't work at all, a multiclass classifcation problem. The model produced managed to get 81.06% accuracy on test data, currently on top 14% of the leaderboeard.

#### 1. Import Dependencies Library
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 
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

"""### 2. Load train & test set data"""

df_train = pd.read_csv("/train_taarifa.csv",parse_dates=True)
y_train = pd.read_csv("/y_train.csv")
df_test = pd.read_csv("/test_taarifa.csv",parse_dates=True)
df_full = pd.concat([df_train,df_test],sort=False).drop('status_group',axis=1)

df_train_index = df_train.index.values
df_test_index = df_test.index.values

print(df_full.shape)
print(df_train.shape)
print(y_train.shape)
print(df_test.shape)

"""### 3. Exploratory Data Analysis"""

df_train.dtypes

# Check if there is any missing data
# There are 7 columns with missing data
df_train.isnull().sum()

# Check the counts of target variable

df_train.status_group.value_counts()

# Plot the bar chart to visualize target "status_group"

palette=[sns.color_palette()[0],sns.color_palette()[2],sns.color_palette()[1]]
y_train.status_group.value_counts().plot(kind='barh', color=palette)

# Plot the bar chart to visualize target "status_group" & relationship with "construction_year"

df_train.construction_year=pd.to_numeric(df_train.construction_year)
df_train.loc[df_train.construction_year <= 0, df_train.columns=='construction_year'] = 1950

hist1=df_train[df_train.status_group == 'functional'].construction_year
hist2=df_train[df_train.status_group == 'functional needs repair'].construction_year
hist3=df_train[df_train.status_group == 'non functional'].construction_year

n,b,p=plt.hist([hist1, hist2, hist3], stacked=True,range=[1950,2010])
plt.legend(['functional','functional needs repair','non functional'],loc=0)
plt.text(1952, 15000,'NO DATA',fontsize=20,rotation=90,color='white')
plt.xlabel('Construction Year', fontsize=18)

#convert status group label into numerical data

val_status_group={'functional':2, 'functional needs repair':1,
                   'non functional':0}
df_train['status_group_vals']=df_train.status_group.replace(val_status_group)

"""##### Remove the field which has too many redundant values/ categories

### 3. Data Transformation - Part 1
"""

# Divide all categorical columns into max 5 categories, else as others
# This will helps in model convergence
def funder_cleaning(train):
    if train['funder']=='Government Of Tanzania':
        return 'government'
    elif train['funder']=='Danida':
        return 'danida'
    elif train['funder']=='Hesawa':
        return 'hesawa'
    elif train['funder']=='Rwssp':
        return 'rwssp'
    elif train['funder']=='World Bank':
        return 'world_bank'    
    else:
        return 'other'
    
df_train['funder']= df_train.apply(lambda row: funder_cleaning(row), axis=1)
df_test['funder']= df_test.apply(lambda row: funder_cleaning(row), axis=1)

def installer_cleaning(train):
    if train['installer']=='DWE':
        return 'dwe'
    elif train['installer']=='Government':
        return 'government'
    elif train['installer']=='RWE':
        return 'rwe'
    elif train['installer']=='Commu':
        return 'commu'
    elif train['installer']=='DANIDA':
        return 'danida'    
    else:
        return 'other'
    
df_train['installer']= df_train.apply(lambda row: installer_cleaning(row), axis=1)
df_test['installer']= df_test.apply(lambda row: installer_cleaning(row), axis=1)

def scheme_wrangler(row):
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

df_train['scheme_management'] = df_train.apply(lambda row: scheme_wrangler(row), axis=1)
df_test['scheme_management'] = df_test.apply(lambda row: scheme_wrangler(row), axis=1)

# Turn construction_year into a categorical column containing the following values: '60s', '70s',
# '80s', '90s, '00s', '10s', 'unknown'.

def construction_wrangler(row):
    if row['construction_year'] >= 1960 and row['construction_year'] < 1970:
        return '60s'
    elif row['construction_year'] >= 1970 and row['construction_year'] < 1980:
        return '70s'
    elif row['construction_year'] >= 1980 and row['construction_year'] < 1990:
        return '80s'
    elif row['construction_year'] >= 1990 and row['construction_year'] < 2000:
        return '90s'
    elif row['construction_year'] >= 2000 and row['construction_year'] < 2010:
        return '00s'
    elif row['construction_year'] >= 2010:
        return '10s'
    else:
        return 'unknown'
    
df_train['construction_year'] = df_train.apply(lambda row: construction_wrangler(row), axis=1)
df_test['construction_year'] = df_test.apply(lambda row: construction_wrangler(row), axis=1)

"""##### Drop the columns which are redundant columns with too many unique values, this would not bring any goods to model"""

# Eg. A few pairs of variables are similar :
# (waterpoint_type,waterpoint_type_group) & (source & source type) & etcs both are almost similar, can drop one of them

df_train=df_train.drop(['subvillage','scheme_name','recorded_by','waterpoint_type','source','quantity',
                        'payment','management','extraction_type','extraction_type_group'],axis=1)
df_test=df_test.drop(['subvillage','scheme_name','recorded_by','waterpoint_type','source','quantity',
                      'payment','management','extraction_type','extraction_type_group'],axis=1)

"""##### Fill the valid columns with blank value - "Unknown"/"Other""""

#Since most of the values are True, as of now lets insert True for the missing values. Scope to alter the values in future
df_train.public_meeting = df_train.public_meeting.fillna('Unknown')
df_test.public_meeting = df_test.public_meeting.fillna('Unknown')

df_train.scheme_management = df_train.scheme_management.fillna('other')
df_test.scheme_management = df_test.scheme_management.fillna('other')

# We only have two values here: true and false. This one can stay but we'll have to replace 
# the unknown data with a string value.

df_train.permit = df_train.permit.fillna('Unknown')
df_test.permit = df_test.permit.fillna('Unknown')

#EDA
# Checking Null Values on test data
df_train.apply(lambda x: sum(x.isnull()), axis=0)

# Statistics for full train/test dataset
df_full.describe()

### Imputation for invalid value 0 in numerical column with mean/median

# amount_tsh median equal to 0, no replacement done in first consideration 

def population_cleaning(train):
    if train['population']==0:
        return 25 #replace with median, skewed data
    else:
        return train['population']

def gpsheight_cleaning(train):
    if train['gps_height']==0:
        return 364 #replace with median, skewed data
    else:
        return train['gps_height']

def longitude_cleaning(train):
    if train['longitude']==0:
        return 34.074262
    else:
        return train['longitude']

df_train['population']= df_train.apply(lambda row: population_cleaning(row), axis=1)
df_test['population']= df_test.apply(lambda row: population_cleaning(row), axis=1)
df_train['gps_height']= df_train.apply(lambda row: gpsheight_cleaning(row), axis=1)
df_test['gps_height']= df_test.apply(lambda row: gpsheight_cleaning(row), axis=1)
df_train['longitude']= df_train.apply(lambda row: longitude_cleaning(row), axis=1)
df_test['longitude']= df_test.apply(lambda row: longitude_cleaning(row), axis=1)

"""### Feature engineering: Days since recorded can be useful to know the period of pump installed"""

df_train.date_recorded = pd.datetime(2013, 12, 3) - pd.to_datetime(df_train.date_recorded)
df_train.columns = ['days_since_recorded' if x == 'date_recorded' else x for x in df_train.columns]
df_train.days_since_recorded = df_train.days_since_recorded.astype('timedelta64[D]').astype(int)

df_test.date_recorded = pd.datetime(2013, 12, 3) - pd.to_datetime(df_test.date_recorded)
df_test.columns = ['days_since_recorded' if x == 'date_recorded' else x for x in df_test.columns]
df_test.days_since_recorded = df_test.days_since_recorded.astype('timedelta64[D]').astype(int)

# Data is clean for model training
# not too correlated to each other, good fit for model
df_train.corr()

#water_quality and quality_group are correlated . drop one of them
df_train=df_train.drop(['quality_group'],axis=1)
df_test=df_test.drop(['quality_group'],axis=1)

# Lastly drop those variable that seem superflous like 'num_private' & 'wpt_name which no extra info
df_train1 = df_train.copy()
df_train = df_train.drop(['num_private','wpt_name','lga','region','ward','status_group'], axis=1)

df_test = df_test.drop(['num_private','wpt_name','lga','region','ward'], axis=1)

"""##### One-hot encoding for categorical data"""

# Get dummy columns for the categorical columns and shuffle the data.

dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management', 'permit',
              'construction_year', 'extraction_type_class','management_group', 'payment_type', 'water_quality',
              'quantity_group', 'source_type', 'source_class','waterpoint_type_group']

df_train = pd.get_dummies(df_train, columns = dummy_cols)

df_train = df_train.sample(frac=1).reset_index(drop=True)

df_test = pd.get_dummies(df_test, columns = dummy_cols)

print('Shape of training data',df_train.shape)
print('Shape of testing data',df_test.shape)

df_train.corr()

#profiling report of training data
#pandas_profiling.ProfileReport(df_train)

#from profiling report, we can drop duplicate rows from the training dataset and also we can see 
#waterpoint_type_group_hand pump is highly correlated with extraction_type_class_handpump. we can drop any one of the column

#train=train.drop_duplicates()

df_train=df_train.drop(['waterpoint_type_group_hand pump'],axis=1)
df_test=df_test.drop(['waterpoint_type_group_hand pump'],axis=1)

df_train=df_train.drop(['source_type_other'],axis=1)
df_test=df_test.drop(['source_type_other'],axis=1)

# Define Y & drop it from the train set for feature selection
target = df_train.status_group_vals
features = df_train.drop('status_group_vals', axis=1)

features1=features.copy()

"""### 4. Feature Selection (Using boruta package)"""

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# RandomForrestClassifier as the estimator to use for Boruta. 
# The max_depth of the tree is advised on the Boruta Github page to be between 3 to 7

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

# number of selected features
print ('\n Number of selected features:')
print (boruta_selector.n_features_)

features1=pd.DataFrame(features.columns.tolist())
features1['rank']=boruta_selector.ranking_
features1 = features1.sort_values('rank', ascending=True).reset_index(drop=True)
print ('\n Top %d features:' % boruta_selector.n_features_)
print (features1.head(boruta_selector.n_features_))

#From Boruta, we retained the 70 columns. The remaining columns whose rank is greater than 1 is not required.

features1=pd.DataFrame(features.columns.tolist())
features1['rank']=boruta_selector.ranking_
features1 = features1.sort_values('rank', ascending=True).reset_index(drop=True)
features1

#Drop columns which Boruta didnt give/predict rank 1 importance

df_train=features.drop(['water_quality_milky'
    ,'source_type_dam'
    ,'payment_type_on failure'
    ,'funder_hesawa'
    ,'construction_year_60s'
    ,'management_group_parastatal'
    ,'water_quality_salty abandoned'
    ,'management_group_unknown'
    ,'management_group_other'
    ,'extraction_type_class_rope pump'
    ,'payment_type_other'
    ,'installer_danida'
    ,'water_quality_fluoride'
    ,'source_class_unknown'
    ,'water_quality_coloured'
    ,'extraction_type_class_wind-powered'
    ,'waterpoint_type_group_cattle trough'
    ,'waterpoint_type_group_dam'
    ,'water_quality_fluoride abandoned'],axis=1)

df_test=df_test.drop(['water_quality_milky'
    ,'source_type_dam'
    ,'payment_type_on failure'
    ,'funder_hesawa'
    ,'construction_year_60s'
    ,'management_group_parastatal'
    ,'water_quality_salty abandoned'
    ,'management_group_unknown'
    ,'management_group_other'
    ,'extraction_type_class_rope pump'
    ,'payment_type_other'
    ,'installer_danida'
    ,'water_quality_fluoride'
    ,'source_class_unknown'
    ,'water_quality_coloured'
    ,'extraction_type_class_wind-powered'
    ,'waterpoint_type_group_cattle trough'
    ,'waterpoint_type_group_dam'
    ,'water_quality_fluoride abandoned'],axis=1)

print(df_train.shape)
print(df_test.shape)

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

# # Check and see if there is missing value now as those invalid value 0 has been replaced with np.nan
# df_train.isnull().sum()

# to test out normalization and scaler
df_train1 = df_train.copy()
df_test1 = df_test.copy()

"""### 5. Split Data into train & validation set"""

X_train, X_val, y_train, y_val = train_test_split(df_train1, target, train_size=0.9, random_state=100)

"""##### Perform Data Transformation - Part 2 (after train_test_split)

##### In this case, IterativeImputer only manage to give accuracy of 80.4% hence will no proceed.
"""

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

# # Then apply transformation to validation and test dataset to avoid data leakage
# col_list = X_train.columns.tolist()
# X_train_mice = pd.DataFrame(X_train_fancy_mice, columns=col_list)
# print(X_train_mice.shape)

# X_train_mice.head()

# # Check and see if there is missing value after MICE imputation
# # The invalid value has been imputed
# X_train_mice.isnull().sum()

# # Apply transformation on validation & test dataset

# X_val_fancy_mice = mice.transform(X_val)
# df_test_fancy_mice = mice.transform(df_test)

# X_val_mice = pd.DataFrame(X_val_fancy_mice, columns=col_list)
# df_test_mice = pd.DataFrame(df_test_fancy_mice, columns=col_list)
# print(X_val_mice.shape)
# print(df_test_mice.shape)

"""##### Data Normalization"""

# Columns intended for normalization
df_num_cols=['amount_tsh','days_since_recorded','longitude','region_code', 'latitude',                      
'district_code','population']

# Plot the distribution of numerical columns

for i, col in enumerate(df_num_cols):
    plt.figure(figsize=(12,5))
    plt.subplot(5, 2, i + 1)
    sns.distplot(XS_train[col])
    plt.title(str(col))
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    plt.tight_layout()
    plt.show()
plt.show()

# # fig=plt.figure(figsize=(20,20))
# for i, col in enumerate(df_full[numeric_col]):
#     # ax = fig.add_subplot(2,6,i+1)
#     plt.figure(i)
#     sns.distplot(df_full[col])

# # fig.tight_layout()
# # plt.show()

# Do normalization and scaling

# Train set normalization
X_train.amount_tsh = X_train.amount_tsh.apply(lambda x: np.log10(x+1))
X_train.days_since_recorded = X_train.days_since_recorded.apply(lambda x: np.log10(x+1))
X_train.longitude = X_train.longitude.apply(lambda x: np.log10(x+1))
X_train.region_code = X_train.region_code.apply(lambda x: np.log10(x+1))
X_train.district_code = X_train.district_code.apply(lambda x: np.log10(x+1))
X_train.population = X_train.population.apply(lambda x: np.log10(x+1))


# Validation set normalization
X_val.amount_tsh = X_val.amount_tsh.apply(lambda x: np.log10(x+1))
X_val.days_since_recorded = X_val.days_since_recorded.apply(lambda x: np.log10(x+1))
X_val.longitude = X_val.longitude.apply(lambda x: np.log10(x+1))
X_val.region_code = X_val.region_code.apply(lambda x: np.log10(x+1))
X_val.district_code = X_val.district_code.apply(lambda x: np.log10(x+1))
X_val.population = X_val.population.apply(lambda x: np.log10(x+1))

# Test set normalization
df_test1.amount_tsh = df_test.amount_tsh.apply(lambda x: np.log10(x+1))
df_test1.days_since_recorded = df_test.days_since_recorded.apply(lambda x: np.log10(x+1))
df_test1.longitude = df_test.longitude.apply(lambda x: np.log10(x+1))
df_test1.region_code = df_test.region_code.apply(lambda x: np.log10(x+1))
df_test1.district_code = df_test.district_code.apply(lambda x: np.log10(x+1))
df_test1.population = df_test.population.apply(lambda x: np.log10(x+1))

df_train1=df_train1.fillna(df_train1.mean())
df_test1=df_test1.fillna(df_test1.mean())

# X_train_mice=X_train_mice.fillna(X_train_mice.mean())
# X_val_mice=X_val_mice.fillna(X_val_mice.mean())

# Data distribution after normalization

for i, col in enumerate(df_num_cols):
    plt.figure(figsize=(12,5))
    plt.subplot(5, 2, i + 1)
    sns.distplot(X_train[col])
    plt.title(str(col))
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    plt.tight_layout()
    plt.show()
plt.show()

"""##### Data Scaling"""

# Scale the numerical variables can help model find pattern and faster convergence
# Tree-like algorithms is still prone to unnormalized/unscaled data due to bagging & boosting
rs =RobustScaler()

# Fit_transform should be done on only training data to avoid data leakage problem
X_train[df_num_cols] = rs.fit_transform(X_train[df_num_cols])

X_val[df_num_cols] = rs.transform(X_val[df_num_cols])
df_test1[df_num_cols] = rs.transform(df_test1[df_num_cols])

"""##### Data are now good to proceed to model training ...

### 6. Model Training

##### Hyperparameter tuning with GridSearchCV
"""

# Model 1 : Linear Kernel SVM Classifier

def Linear_svc_model(X_train, X_val, y_train, y_val):
    if __name__ == '__main__':
        
        #scl = StandardScaler()
        clf = LinearSVC()
        
        parameters = {'C':[0.001,0.01,0.1,1.0,10.0,100.0],'class_weight':[None, 'balanced']}

        estimator = GridSearchCV(clf, parameters,n_jobs=-1)

        estimator.fit(X_train, y_train)

        best_params = estimator.best_params_
                                 
        validation_accuracy = estimator.score(X_val, y_val)
        print('Validation accuracy: ', validation_accuracy)
        print(best_params)

Linear_svc_model(X_train, X_val, y_train, y_val)

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

GradientBoostingClassifier_model(X_train, X_val, y_train, y_val)

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

RandomForestClassifier_model(X_train, X_val, y_train, y_val)

rf = RandomForestClassifier(
                      max_samples = 0.8,
                      max_depth= 22,
                    #   n_jobs=1,
                    #   oob_score=True,
                      n_estimators= 1000)
 

rf.fit(X_train, y_train)

rf.score(X_val, y_val)

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

# Model 4 : XGBoost

def XGBoost_model(X_train, X_val, y_train, y_val):
    if __name__ == '__main__':
        
        
        xgboost = xgb.XGBClassifier()
        
        parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                      'objective':['multi:softprob'],
                      'num_class':[3],
                      'eta': [0.05],
                      'max_depth': [10,15],
                    #   'lambda' : [0.01],
                      'alpha' : [0.1],
                      'colsample_bylevel': [0.6],
                    #   'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                      'colsample_bytree': [0.7],
                    #   'missing':[-999],
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

XGBoost_model(X_train, X_val, y_train, y_val)

#prod

xgbf = xgb.XGBClassifier(nthread=4, #when use hyperthread, xgboost may become slower
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
                    #   'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                      colsample_bytree=0.7,
                    #   'missing':[-999],
                      seed=103734,
                      n_estimators=100)
        

xgbf.fit(X_train, y_train)

xgbf.score(X_val,y_val)

"""### Final Model for submission : Random Forest (Validation Accuracy : 0.8225)"""

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