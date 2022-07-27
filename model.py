import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import wrangle
import explore
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer,PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error

def split(df):    
    train, test = train_test_split(df, random_state=123)
    train, validate = train_test_split(train,random_state=123)
    return(train,validate,test)

def model_split(train,validate,test):
    x_train = train.drop(columns='tax_value')
    y_train = train.tax_value

    x_validate = validate.drop(columns='tax_value')
    y_validate = validate.tax_value

    x_test = test.drop(columns='tax_value')
    y_test = test.tax_value
    
    return(x_train,y_train,x_validate,y_validate,x_test,y_test)
def lasso_test(df):
    #split the data
    train,validate,test = split(df)
    x_train, y_train,x_validate,y_validate,x_test,y_test = model_split(train,validate,test)
    
    #combine bed and bath
    x_train['bed_bath'] = x_train.bedrooms + x_train.bathrooms
    x_train=x_train.drop(columns=['bedrooms','bathrooms'])
    x_validate['bed_bath'] = x_validate.bedrooms + x_validate.bathrooms
    x_validate=x_validate.drop(columns=['bedrooms','bathrooms'])
    x_test['bed_bath'] = x_test.bedrooms + x_test.bathrooms
    x_test=x_test.drop(columns=['bedrooms','bathrooms'])
    
    # scale the data
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_train_scaled = pd.DataFrame(x_train_scaled)
    x_train_scaled.columns = x_train.columns.tolist()
    
    x_validate_scaled = scaler.transform(x_validate)
    x_validate_scaled = pd.DataFrame(x_validate_scaled)
    x_validate_scaled.columns = x_train_scaled.columns.tolist()

    x_test_scaled = scaler.transform(x_test)
    x_test_scaled = pd.DataFrame(x_test_scaled)
    x_test_scaled.columns = x_train_scaled.columns.tolist()
    # create a model object for the RFE
    lm = LinearRegression()
    # select the most useful features
    rfe = RFE(lm, n_features_to_select=5)
    rfe.fit(x_train_scaled,y_train)
    feature_mask = rfe.support_
    rfe_feature = x_train_scaled.iloc[:,feature_mask].columns.tolist()
    print(f'The 5 most useful features for this county are {rfe_feature}',
         '\n---------------')
    
    # limit data frame to only the most useful features
    x_train_scaled= x_train_scaled[rfe_feature]
    x_validate_scaled= x_validate_scaled[rfe_feature]
    x_test_scaled= x_test_scaled[rfe_feature]
    
    # Change series into data frame
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    lars = LassoLars(alpha=1.0)
    lars.fit(x_train_scaled, y_train.tax_value)

    # predict test
    y_test['pred_lars'] = lars.predict(x_test_scaled)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.tax_value, y_test.pred_lars)**(1/2)


    print("RMSE for Lasso + Lars\nTest/Out-of-Sample: ", rmse_test,
          f'\nmodel performed {(221854.16-rmse_test)/221854.16 :.2%} better than baseline',
         "\n-----------------")



def poly_test(df):
    #split the data
    train,validate,test = split(df)
    x_train, y_train,x_validate,y_validate,x_test,y_test = model_split(train,validate,test)
    
    #combine bed and bath
    x_train['bed_bath'] = x_train.bedrooms + x_train.bathrooms
    x_train=x_train.drop(columns=['bedrooms','bathrooms'])
    x_validate['bed_bath'] = x_validate.bedrooms + x_validate.bathrooms
    x_validate=x_validate.drop(columns=['bedrooms','bathrooms'])
    x_test['bed_bath'] = x_test.bedrooms + x_test.bathrooms
    x_test=x_test.drop(columns=['bedrooms','bathrooms'])
    
    # scale the data
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_train_scaled = pd.DataFrame(x_train_scaled)
    x_train_scaled.columns = x_train.columns.tolist()
    
    x_validate_scaled = scaler.transform(x_validate)
    x_validate_scaled = pd.DataFrame(x_validate_scaled)
    x_validate_scaled.columns = x_train_scaled.columns.tolist()

    x_test_scaled = scaler.transform(x_test)
    x_test_scaled = pd.DataFrame(x_test_scaled)
    x_test_scaled.columns = x_train_scaled.columns.tolist()
    # create a model object for the RFE
    lm = LinearRegression()
    # select the most useful features
    rfe = RFE(lm, n_features_to_select=5)
    rfe.fit(x_train_scaled,y_train)
    feature_mask = rfe.support_
    rfe_feature = x_train_scaled.iloc[:,feature_mask].columns.tolist()
    print(f'The 5 most useful features for this county are {rfe_feature}',
         '\n---------------')
    
    # limit data frame to only the most useful features
    x_train_scaled= x_train_scaled[rfe_feature]
    x_validate_scaled= x_validate_scaled[rfe_feature]
    x_test_scaled= x_test_scaled[rfe_feature]
    
    # Change series into data frame
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    pf = PolynomialFeatures(degree=4)
    x_train_degree2 = pf.fit_transform(x_train_scaled)
    x_validate_degree2 = pf.transform(x_validate_scaled)
    x_test_degree2 = pf.transform(x_test_scaled)
    lm2 = LinearRegression(normalize=True )
    lm2.fit(x_train_degree2, y_train.tax_value)
    y_test['pred_lm2'] = lm2.predict(x_test_degree2)
    rmse_test = mean_squared_error(y_test.tax_value, y_test.pred_lm2)**(1/2)

    
    print("RMSE for Polynomial Model, degrees=4"
      "\nTest/Out-of-Sample: ", rmse_test,
          f'\nmodel performed {(221854.16-rmse_test)/221854.16 :.2%} better than baseline',
         "\n-----------------")


def models(df):
    #split the data
    train,validate,test = split(df)
    x_train, y_train,x_validate,y_validate,x_test,y_test = model_split(train,validate,test)
    
    #combine bed and bath
    x_train['bed_bath'] = x_train.bedrooms + x_train.bathrooms
    x_train=x_train.drop(columns=['bedrooms','bathrooms'])
    x_validate['bed_bath'] = x_validate.bedrooms + x_validate.bathrooms
    x_validate=x_validate.drop(columns=['bedrooms','bathrooms'])
    x_test['bed_bath'] = x_test.bedrooms + x_test.bathrooms
    x_test=x_test.drop(columns=['bedrooms','bathrooms'])
    
    # scale the data
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_train_scaled = pd.DataFrame(x_train_scaled)
    x_train_scaled.columns = x_train.columns.tolist()
    
    x_validate_scaled = scaler.transform(x_validate)
    x_validate_scaled = pd.DataFrame(x_validate_scaled)
    x_validate_scaled.columns = x_train_scaled.columns.tolist()

    x_test_scaled = scaler.transform(x_test)
    x_test_scaled = pd.DataFrame(x_test_scaled)
    x_test_scaled.columns = x_train_scaled.columns.tolist()
    # create a model object for the RFE
    lm = LinearRegression()
    # select the most useful features
    rfe = RFE(lm, n_features_to_select=5)
    rfe.fit(x_train_scaled,y_train)
    feature_mask = rfe.support_
    rfe_feature = x_train_scaled.iloc[:,feature_mask].columns.tolist()
    print(f'The 5 most useful features for this county are {rfe_feature}',
         '\n---------------')
    
    # limit data frame to only the most useful features
    x_train_scaled= x_train_scaled[rfe_feature]
    x_validate_scaled= x_validate_scaled[rfe_feature]
    x_test_scaled= x_test_scaled[rfe_feature]
    
    # Change series into data frame
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    # compare baseline as mean vs baseline as median
    baseline_mean = round(y_train.tax_value.mean(),2)
    baseline_median = y_train.tax_value.median()

    y_train['baseline_mean'] = baseline_mean
    y_train['baseline_median'] = baseline_median

    y_validate['baseline_mean'] = baseline_mean
    y_validate['baseline_median'] = baseline_median
    
    
    #calculate errors for baseline
    rmse_train = mean_squared_error(y_train.tax_value, y_train.baseline_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.baseline_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2),
         "\n-----------------")

    rmse_train = mean_squared_error(y_train.tax_value, y_train.baseline_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.baseline_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2),
         "\n-----------------")
    
    
    # create the model object
    lm = LinearRegression(normalize=True)


    lm.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['pred_lm'] = lm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lm)**(1/2)

    # predict validate
    y_validate['pred_lm'] = lm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate,
          f'\nmodel performed {(221854.16-rmse_train)/221854.16 :.2%} better than baseline',
         "\n-----------------")

    
    # create the model object
    lars = LassoLars(alpha=1.0)
    lars.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['pred_lars'] = lars.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lars)**(1/2)

    # predict validate
    y_validate['pred_lars'] = lars.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate,
          f'\nmodel performed {(221854.16-rmse_train)/221854.16 :.2%} better than baseline',
         "\n-----------------")

    
    
    pf = PolynomialFeatures(degree=4)

    # fit and transform X_train_scaled
    x_train_degree2 = pf.fit_transform(x_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    x_validate_degree2 = pf.transform(x_validate_scaled)
    x_test_degree2 = pf.transform(x_test_scaled)

    
    # create the model object
    lm2 = LinearRegression(normalize=True )

    lm2.fit(x_train_degree2, y_train.tax_value)

    # predict train
    y_train['pred_lm2'] = lm2.predict(x_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lm2)**(1/2)

    # predict validate

    y_validate['pred_lm2'] = lm2.predict(x_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=4\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate,
          f'\nmodel performed {(221854.16-rmse_train)/221854.16 :.2%} better than baseline',
         "\n-----------------")
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    glm.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['pred_glm'] = glm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_glm)**(1/2)

    # predict validate
    y_validate['pred_glm'] = glm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining: ", rmse_train, 
      "\nValidation: ", rmse_validate,
          f'\nmodel performed {(221854.16-rmse_train)/221854.16 :.2%} better than baseline',
         )
    
    
    plt.figure(figsize=(16,8))
    plt.plot(y_validate.tax_value, y_validate.baseline_mean, alpha=1, color="red", label='_nolegend_')
    
    plt.plot(y_validate.tax_value, y_validate.tax_value, alpha=1, color="orange", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5))

    plt.scatter(y_validate.tax_value, y_validate.pred_lm, 
            alpha=.5, color="red", s=10, label="Model: LinearRegression")
    plt.scatter(y_validate.tax_value, y_validate.pred_lars, 
            alpha=.5, color="magenta", s=10, label="Model: LinearRegression")
    plt.scatter(y_validate.tax_value, y_validate.pred_glm, 
            alpha=.5, color="blue", s=10, label="Model: TweedieRegressor")
    plt.scatter(y_validate.tax_value, y_validate.pred_lm2, 
            alpha=.5, color="green", s=10, label="Model 4th degree Polynomial")
    plt.legend()
    plt.xlabel("Actual Tax Value")
    plt.ylabel("Predicted Tax Value")
    plt.title("Where are predictions closest to the actual?")
    plt.show()

def big_model(df):
    #split the data
    df = df.drop(columns='fips')
    train,validate,test = split(df)
    x_train, y_train,x_validate,y_validate,x_test,y_test = model_split(train,validate,test)
    
    #combine bed and bath
    x_train['bed_bath'] = x_train.bedrooms + x_train.bathrooms
    x_train=x_train.drop(columns=['bedrooms','bathrooms'])
    x_validate['bed_bath'] = x_validate.bedrooms + x_validate.bathrooms
    x_validate=x_validate.drop(columns=['bedrooms','bathrooms'])
    x_test['bed_bath'] = x_test.bedrooms + x_test.bathrooms
    x_test=x_test.drop(columns=['bedrooms','bathrooms'])
    
    # scale the data
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_train_scaled = pd.DataFrame(x_train_scaled)
    x_train_scaled.columns = x_train.columns.tolist()
    
    x_validate_scaled = scaler.transform(x_validate)
    x_validate_scaled = pd.DataFrame(x_validate_scaled)
    x_validate_scaled.columns = x_train_scaled.columns.tolist()

    x_test_scaled = scaler.transform(x_test)
    x_test_scaled = pd.DataFrame(x_test_scaled)
    x_test_scaled.columns = x_train_scaled.columns.tolist()
    # create a model object for the RFE
    lm = LinearRegression()
    # select the most useful features
    rfe = RFE(lm, n_features_to_select=5)
    rfe.fit(x_train_scaled,y_train)
    feature_mask = rfe.support_
    rfe_feature = x_train_scaled.iloc[:,feature_mask].columns.tolist()
    print(f'The 5 most useful features for this county are {rfe_feature}',
         '\n---------------')
    
    # limit data frame to only the most useful features
    x_train_scaled= x_train_scaled[rfe_feature]
    x_validate_scaled= x_validate_scaled[rfe_feature]
    x_test_scaled= x_test_scaled[rfe_feature]
    
    # Change series into data frame
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    # compare baseline as mean vs baseline as median
    baseline_mean = round(y_train.tax_value.mean(),2)
    baseline_median = y_train.tax_value.median()

    y_train['baseline_mean'] = baseline_mean
    y_train['baseline_median'] = baseline_median

    y_validate['baseline_mean'] = baseline_mean
    y_validate['baseline_median'] = baseline_median
    
    
    #calculate errors for baseline
    rmse_train = mean_squared_error(y_train.tax_value, y_train.baseline_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.baseline_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2),
         "\n-----------------")

    rmse_train = mean_squared_error(y_train.tax_value, y_train.baseline_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.baseline_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2),
         "\n-----------------")
    
    
    # create the model object
    lm = LinearRegression(normalize=True)


    lm.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['pred_lm'] = lm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lm)**(1/2)

    # predict validate
    y_validate['pred_lm'] = lm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate,
          f'\nmodel performed {(221854.16-rmse_train)/221854.16 :.2%} better than baseline',
         "\n-----------------")

    
    # create the model object
    lars = LassoLars(alpha=1.0)
    lars.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['pred_lars'] = lars.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lars)**(1/2)

    # predict validate
    y_validate['pred_lars'] = lars.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate,
          f'\nmodel performed {(221854.16-rmse_train)/221854.16 :.2%} better than baseline',
         "\n-----------------")

    
    
    pf = PolynomialFeatures(degree=4)

    # fit and transform X_train_scaled
    x_train_degree2 = pf.fit_transform(x_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    x_validate_degree2 = pf.transform(x_validate_scaled)
    x_test_degree2 = pf.transform(x_test_scaled)

    
    # create the model object
    lm2 = LinearRegression(normalize=True )

    lm2.fit(x_train_degree2, y_train.tax_value)

    # predict train
    y_train['pred_lm2'] = lm2.predict(x_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lm2)**(1/2)

    # predict validate

    y_validate['pred_lm2'] = lm2.predict(x_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=4\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate,
          f'\nmodel performed {(221854.16-rmse_train)/221854.16 :.2%} better than baseline',
         "\n-----------------")
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    glm.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['pred_glm'] = glm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_glm)**(1/2)

    # predict validate
    y_validate['pred_glm'] = glm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining: ", rmse_train, 
      "\nValidation: ", rmse_validate,
          f'\nmodel performed {(221854.16-rmse_train)/221854.16 :.2%} better than baseline',
         )
    
    
    plt.figure(figsize=(16,8))
    plt.plot(y_validate.tax_value, y_validate.baseline_mean, alpha=1, color="red", label='_nolegend_')
    
    plt.plot(y_validate.tax_value, y_validate.tax_value, alpha=1, color="orange", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5))

    plt.scatter(y_validate.tax_value, y_validate.pred_lm, 
            alpha=.5, color="red", s=10, label="Model: LinearRegression")
    plt.scatter(y_validate.tax_value, y_validate.pred_lars, 
            alpha=.5, color="magenta", s=10, label="Model: LinearRegression")
    plt.scatter(y_validate.tax_value, y_validate.pred_glm, 
            alpha=.5, color="blue", s=10, label="Model: TweedieRegressor")
    plt.scatter(y_validate.tax_value, y_validate.pred_lm2, 
            alpha=.5, color="green", s=10, label="Model 4th degree Polynomial")
    plt.legend()
    plt.xlabel("Actual Tax Value")
    plt.ylabel("Predicted Tax Value")
    plt.title("Where are predictions closest to the actual?")
    plt.show()
