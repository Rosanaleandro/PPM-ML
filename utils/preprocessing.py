from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer

#from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss



def convert_type(df):
    '''
        Converte o tipo de uma lista de tipos para formato de string
            --------
    paramers:
        df: dataset for conversion
        
        return: list
    '''  
  
    lst=[]
  
    for column in df:
        if df[column].dtype=='float64': 
            lst.append('float64')
        elif df[column].dtype=='int64': 
            lst.append('int64') 
        elif df[column].dtype=='object': 
            lst.append('String')
        else:
            lst.append('bolleano')
    return lst

def convert_type_column(df, coluna):
    '''
        Converte o tipo de uma lista de tipos para formato de string
            --------
    paramers:
        df: dataset for conversion
        coluna : column to be converted
        
        return: string
    '''  

    if df[coluna].dtype=='float64': 
        return 'float64'
    elif df[coluna].dtype=='int64': 
        return 'int64'
    elif df[coluna].dtype=='object': 
        return 'String'
    else:
        return 'bolleano'


def binning(df, n_bins=None, encode=None, strategy=None):
    
    data = df.copy()
    
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
            discretizer.fit(data[col].values.reshape(-1, 1))
            data[col] = discretizer.transform(data[col].values.reshape(-1, 1))
        
        return data
    elif isinstance(data, pd.Series):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        
        col_name = data.name
        discretizer.fit(data.values.reshape(-1, 1))
        data = discretizer.transform(data.values.reshape(-1, 1))
        
        dt = pd.DataFrame(data, columns=[col_name])
        
        return dt


def scaling(df, target_name=None):
    '''
    Objetiva a aplicação de normalização em colunas numéricas
  
    --------
    parameters:

        data:colunas para transformação
        type: pandas.DataFrame
        
        target_name: nome da coluna target
        type: str
        
    return pandas.DataFrame
    '''
    
    data = df.copy()
    
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if col not in target_name:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(data[col].values.reshape(-1, 1))
                data[col] = scaler.transform(data[col].values.reshape(-1, 1))
        
        return data
    elif isinstance(data, pd.Series):
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        col_name = data.name
        scaler.fit(data.values.reshape(-1, 1))
        data = scaler.transform(data.values.reshape(-1, 1))
        
        dt = pd.DataFrame(data, columns=[col_name])
        
        return dt
        


def standardization(df, target_name=None):
    '''
    Objetiva a aplicação de padronização em colunas numéricas
    
    --------
    parameters:

        data:atributos para transformação
        type: pandas.DataFrame
        
        target_name: Nome da variável target
        type; str
        
    return pandas.DataFrame
    '''
  
    data = df.copy()
    
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if col not in target_name:
                scaler = StandardScaler()
                scaler.fit(data[col].values.reshape(-1, 1))
                data[col] = scaler.transform(data[col].values.reshape(-1, 1))
        
        return data
    elif isinstance(data, pd.Series):
        scaler = StandardScaler()
        
        col_name = data.name
        scaler.fit(data.values.reshape(-1, 1))
        data = scaler.transform(data.values.reshape(-1, 1))
        
        dt = pd.DataFrame(data, columns=[col_name])
        
        return dt


def onehot_encoder(df,coluna=None):
    '''
    Aplica onehot encoder em colunas categóricas
    
    --------
    parameters:

        data:colunas para tranformação
        type: pandas.DataFrame
        
    return pandas.DataFrame
    '''
    
    data = df.copy()
   
    transform = pd.get_dummies(data,  columns=coluna, drop_first = True)
   
    return transform


def ordinal_encoder(df):
    '''
    Aplica ordinal encoder em colunas categóricas
    
    --------
    paramers:

        data:colunas para transformação
        type: pandas.DataFrame
        
    return pandas.DataFrame
    '''
  
    data = df.copy()
    
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(data)
    transform = ordinal_encoder.transform(data)
    
    return transform


def over_sampling(X, y):
    '''
    Aplica a técnica oversampling method SMOTE para balanceamento de classes
    
    --------
    parameters:

        X: colunas independentes
        type: numpy.array 2D
        
        y: dependent column
        type: numpy.array 1D
        
    return pandas.DataFrame
    '''
    
    oversampling = SMOTE()
    X_resampled, y_resampled = oversampling.fit_resample(X, y)
    
    return X_resampled, y_resampled

def under_sampling(X, y):
    '''
    Aplica a técnica undersampling para balancear as classes
    
    --------
    paramers:

        X: independent columns
        type: numpy.array 2D
        
        y: dependent column
        type: numpy.array 1D
        
    return pandas.DataFrame
    '''
    undersample = NearMiss()
    X_resampled, y_resampled = undersample.fit_resample(X, y)
    
    return X_resampled, y_resampled
