import pandas as pd
import os
from scipy import stats
from datetime import date
from env import username, host, password
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split





def bed_bath_test(df):
    bedrooms = df.bedrooms
    bathrooms = df.bathrooms
    print(bedrooms.var())
    print(bathrooms.var())
    t, p = stats.ttest_ind(bedrooms, bathrooms)
    if p <.05:
        df['bed_bath'] = df.bedrooms + df.bathrooms
        print(f'Our confidence is {(1-p):.0%} that the variables are dependent. We reject the null hypothesis.')
    else:
        print('we cannot reject null hypothesis')
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.hist(bedrooms,bins=5)
    plt.title('Bedrooms')
    plt.subplot(122)
    plt.hist(bathrooms,bins=5)
    plt.title('bathrooms')
    plt.show


def split(df):    
    train, test = train_test_split(df, random_state=123)
    train, validate = train_test_split(train)
    return(train,validate,test)

