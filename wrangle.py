import pandas as pd
import os
from datetime import date
from env import username, host, password
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import explore
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def get_connection(db, user=username, host=host, password=password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
   
    sql_query = """
             
    select bedroomcnt as 'bedrooms' , 
	    bathroomcnt as 'bathrooms' , 
        buildingqualitytypeid as 'quality',
	    calculatedfinishedsquarefeet as 'sq_feet', 
        poolcnt as 'pool',
	    taxvaluedollarcnt as 'tax_value', 
	    yearbuilt, 
	    fips  from properties_2017
    join predictions_2017 as p using(parcelid)
    WHERE propertylandusetypeid = 261
    and p.transactiondate = 2017;

                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def get_zillow_data():
    
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv') 
      
    return df

def handle_nulls(df):
    df = df.dropna()
    return(df)

def optimize_types(df):
    cols = ['bedrooms',
            'yearbuilt',
            ]

    df[cols] = df[cols].astype(int)
    df.fips = (df.fips).astype(int)
    df.fips = df.fips.astype(object)
    return(df)


def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
    df = df[(df.bathrooms <= 6) ]
    
    df = df[(df.bedrooms <= 6) & (df.bedrooms >0)]

    df = df[df.tax_value < 1_000_000]
    #df = df[df.sq_feet < 7000]
    return df


def wrangle_zillow():
    """
    Acquires Zillow data
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    returns a clean dataframe
    """
    df = get_zillow_data()

    print(f'There are {len(df)} different homes in this data set')
    
    df = handle_outliers(df)

    df.fips = (df.fips).astype('object')

    df.pool = df.pool.replace(np.NaN, 0)

    imp_mode = IterativeImputer(initial_strategy='most_frequent')
    imp_mode.fit(df)
    tdf = imp_mode.transform(df)
    df = pd.DataFrame(tdf)
    df.columns = ['bedrooms','bathrooms','quality','sq_feet','pool','tax_value','yearbuilt','fips']
    
    df.fips = df.fips.replace([6037,6059,6111],['los_angeles','orange','ventura'])

    df.yearbuilt = df.yearbuilt.round()
    df['age'] =  date.today().year - df.yearbuilt
    df = df.drop(columns = 'yearbuilt')
    
    df.quality = df.quality.round()


    
    return(df)


def county_split(df):
    la = df[df.fips=='los_angeles']
    orange = df[df.fips=='orange']
    ventura = df[df.fips=='ventura']

    la = la.drop(columns='fips')
    orange = orange.drop(columns='fips')
    ventura = ventura.drop(columns='fips')
    return(la, orange, ventura)

