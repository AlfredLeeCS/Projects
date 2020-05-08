# -*- coding: utf-8 -*-
"""MultiClass_Classification_for_Taarifa_WaterPump (CatBoost).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YSnNhqSUv9VTtyvYKSTx_PcF5TLm-V73

### Project Description : This is competition hosted by Data Driven. The objective is to predict which water pumps are functional, which need some repairs, and which don't work at all, a multiclass classifcation problem. The model produced managed to get 81.06% accuracy on test data, currently on top 14% of the leaderboeard.

##### This training use CatBoost Model, under family of boosting tree model, invented in year 2017 by a Russian company, Yandex. It is famous for its power and ability to do categorical features preprocessing directly, unlike other memebrs in the gradient boosting family which make the essence of CatBoost model.

#### 1. Import Dependencies Library
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 
import seaborn as sns
import xgboost as xgb
import catboost

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
from catboost import Pool, CatBoostClassifier

"""### 2. Load train & test set data"""

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

##### Fill the blank categorical column with "Unknown"/"Other"
"""

df_train.funder = df_train.funder.fillna('other')
df_test.funder = df_test.funder.fillna('other')

df_train.installer = df_train.installer.fillna('other')
df_test.installer = df_test.installer.fillna('other')

df_train.scheme_management = df_train.scheme_management.fillna('other')
df_test.scheme_management = df_test.scheme_management.fillna('other')

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
df_train = df_train.drop(['num_private','wpt_name','status_group'], axis=1)

df_test = df_test.drop(['num_private','wpt_name'], axis=1)

print('Shape of training data',df_train.shape)
print('Shape of testing data',df_test.shape)

# Define Y & drop it from the train set for feature selection
target = df_train.status_group_vals
df_train = df_train.drop('status_group_vals', axis=1)

df_train1=df_train.copy()

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

# to test out normalization and scaler
df_train1 = df_train.copy()
df_test1 = df_test.copy()

"""### 5. Split Data into train & validation set"""

X_train, X_val, y_train, y_val = train_test_split(df_train1, target, train_size=0.9,random_state=100)

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
    sns.distplot(X_train[col])
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
"""

# Need to pass in the index of categorical features, elase they will be treated as numeric

cat_features = [3,5,8,9,12,13,15,16,17,18,19,20,21,22,23,24,25,26]

cbc = CatBoostClassifier(
                         loss_function='MultiClass',
                         eval_metric='Accuracy', # or"AUC"
                         leaf_estimation_method='Newton',
                         random_strength=0.1, #0.1
                         learning_rate=0.07,
                         random_seed=103734,
                        #  subsample=1.0,
                         grow_policy='Depthwise', 
                        #  class_weights=[0.39,0.07,0.54],
                         rsm=0.6,
                         classes_count=3,
                         min_data_in_leaf=2,
                        #  langevin=True,
                         l2_leaf_reg = 3, #0.01
                         bagging_temperature=1, #bootstrap aggressiveness
                        #  leaf_estimation_iterations=5, #another parameter for faster training time
                         depth=8, #15
                         border_count=254,
                         verbose=True
                        #  iterations=100,
                        #  one_hot_max_size=31, #bigger reduce training time, >this thres change to onehot
                        #  max_ctr_complexity=2, #reduce trraining time to only produce two pattern combination
                         )
        

cbc.fit(X_train, 
        y_train,
        cat_features=cat_features,
        logging_level='Verbose',
        eval_set=(X_val, y_val),
        # early_stopping_rounds=100,
        use_best_model=True,
        plot=True)

print("Count of trees in model = {}".format(cbc.tree_count_))

# Print Feature Importance

train_pool = Pool(X_train, y_train, cat_features=cat_features)
feature_importances = cbc.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))

# Plotting

sns.set(font_scale=2)

def func_plot_importance(df_imp):

    sns.set(font_scale=1)
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = sns.barplot(
        x="Importance", y="Features", data=df_imp, label="Total", color="b")
    ax.tick_params(labelcolor='k', labelsize='10', width=3)
    plt.show()

def display_importance(model_out, columns, printing=True, plotting=True):
    importances = model_out.feature_importances_
    indices = np.argsort(importances)[::-1]
    importance_list = []
    for f in range(len(columns)):
        importance_list.append((columns[indices[f]], importances[indices[f]]))
        if printing:
            print("%2d) %-*s %f" % (f + 1, 30, columns[indices[f]],
                                    importances[indices[f]]))
    if plotting:
        df_imp = pd.DataFrame(
            importance_list, columns=['Features', 'Importance'])
        func_plot_importance(df_imp)
        

display_importance(model_out=cbc, columns=X_train.columns)

cbc.best_iteration_, cbc.best_score_, cbc.tree_count_

# Training & Validation accuracy

from sklearn import metrics

def auc2(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train),multi_class="ovr"),
                            metrics.roc_auc_score(y_val,m.predict_proba(test),multi_class='ovr'))
auc2(cbc, X_train, X_val)

"""### Final Model for submission :"""

catboost_pred=cbc.predict(df_test1)
test_df = pd.DataFrame(catboost_pred,columns=['status_group'])

test_df['status_group'].value_counts()

predict_cbc1 = pd.concat([test_df, df_test1], axis=1)

predict_cbc1a = predict_cbc1[['id','status_group']]
predict_cbc1a = predict_cbc1a.astype({"id":'object', "status_group":'object'}) 

def parse_values(x):
    if x == 2: 
        return 'functional'
    elif x == 1: 
        return 'functional needs repair'
    elif x == 0: 
        return 'non functional'

predict_cbc1a['status_group'] = predict_cbc1a['status_group'].apply(parse_values)

predict_cbc1a = predict_cbc1a.reset_index(drop=True).set_index('id')

predict_cbc1a.to_csv('pred_cbc4.csv') 

!cp pred_cbc4.csv "drive/My Drive/Data_Interchange"