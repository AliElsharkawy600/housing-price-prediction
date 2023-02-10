#!/usr/bin/env python
# coding: utf-8



# major lab
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector




FILE_PATH = os.path.join(os.getcwd(),'housing.csv')
df_housing = pd.read_csv(FILE_PATH)

df_housing['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN',inplace=True)


df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
df_housing['bedrooms_per_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']
df_housing['population_per_households'] = df_housing['population'] / df_housing['households']


X = df_housing.drop(columns='median_house_value', axis=1)
y = df_housing['median_house_value']

## split to train_full and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)




num_colms = [col for col in x_train.columns if x_train[col].dtype  in ['int32', 'int64', 'float32' ,'float64']]
cat_colms = [col for col in x_train.columns if x_train[col].dtype not in ['int32', 'int64', 'float32' ,'float64']]




# categoricl pipeline
cat_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(cat_colms)),
                                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('ohe', OneHotEncoder(sparse=False))
                            ])

#numerical pipeline
num_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(num_colms)),
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
                            ])

total_pipeline = FeatureUnion(transformer_list=[
                                                ('num', num_pipeline),
                                                ('categ', cat_pipeline)
                                            ])

x_train_final = total_pipeline.fit_transform(x_train)



def preprocess_new(x_new):
    ''' This Function tries to process the new instances before predicted using Model
    Args:
    *****
        (X_new: 2D array) --> The Features in the same order
                ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
                 'population', 'households', 'median_income', 'ocean_proximity']
        All Featutes are Numerical, except the last one is Categorical.
        
     Returns:
     *******
         Preprocessed Features ready to make inference by the Model
    '''
    return total_pipeline.transform(x_new)




preprocess_new(x_test.iloc[10 : 500])

