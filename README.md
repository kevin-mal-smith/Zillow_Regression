# What Is My House Worth?
---
by Kevin smith 7/26/2022

## Project Goal
---
The goal of this project is to develop a home price estimation model that performs better than the baseline prediction, and develop recommendations for ways that the model can be improved and deployed. 

This goal will be accomplished utilizing the following steps:

* Planning
* Acqusition
* Prep
* Exploration
* Feature Engineering
* Modeling
* Delivery

### Steps For Reproduction
---
1. You will need an <mark>env.py</mark> file that contains the hostname, username and password of the mySQL database that contains the <mark>telco_churn</mark> database. Store that env file locally in the repository.
2. Clone my repo (including the <mark>acquire.py</mark> , <mark>prepare.py</mark> & <mark>wrangle.py</mark>files.
3. The libraries used are pandas, numpy, scipy, matplotlib, seaborn, and sklearn.
4. You should now be able to run the <mark>zillow_final_report.ipynb</mark> file.

## Planning
---
Their are two essential parts to any good plan. Identify your **Goals**, and the necessary **Steps** to get there. 

### Goals:
1. Identify variables driving housing prices.
2. Develop a model to make value predicitons based on those variable. 
3. Deliver actionable takeaways

### Steps:
1. Initial hypothesis
2. Acquire and cache the dataset
3. Clean, prep, and split the data to prevent data leakage
4. Do some preliminary exploration of the data (including visualiztions and statistical analyses)*
5. Trim dataset of variables that are not statistically significant
6. Determine which machine learning model perfoms the best
7. Utilize the best model on the test dataset
8. Create a final report notebook with streamlined code optimized for a technical audience

*at least 4 visualizations and 2 statistical analyses

## Data Library
---
| **Variable Name** | **Explanation** | **Values** |
| :---: | :---: | :---: |
| bedrooms | The number of bedrooms in the house | UNumeric value |
| bathrooms | The number of bathrooms in the house | Numeric value |
| quality | a numeric score based on quality on construction | Numeric value|
| sq_feet | The total area inside the home | Numeric value |
| pool | Whether or not the house has a pool | Yes=1/No=0|
| tax_value| The taxable value of the home in $USD | Numeric |
| yearbuilt | The year in which the home was originally built | Year |
| fips | A unique code specific to the county in which the home is located| Numeric |



## Initial Hypothesis
--- 
The initial hypothesis can be based on a gut instinct or the first question that comes to mind when encountering a dataset.

|**Initial hypothesis number** |**hypothesis** |
| :---: | :---: |
|Initial hypothesis 1 | Square footage drives up home value|
|Initial hypothesis 2 | Age drives down home value|

## Acquire and Cache
---
Utitlize the functions imported from the <mark>acquire.py</mark> to create a DataFrame with pandas.

These functions will also cache the data to reduce execution time in the future should we need to create the DataFrame again.

## Prep
--- 
In this step we will utilize the functions in the <mark>wrangle.py</mark> file to get our data ready for exploration. 

This means that we will be looking for columns that may be dropped because they are duplicates, and either dropping or filling any rows that contain blanks depending on the number of blank rows there are.

This also means that we will be splitting the data into 3 separate DataFrames in order to prevent data leakage corrupting our exploration and modeling phases.


## Exploration
---
This is the fun part! this is where we get to ask questions, form hypothesss based on the answers to those questions and use our skills as data scientist to evaluate those hypotheses!

For example, in the Telco dataset I asked "Do people who pay more, churn more?" and unsurprisingly the answer was generally yes. This lead me to the hypothesis that churn would have a dependent relationship with monthly charges, which hypothesis testing confirmed. However I was able to find 3 other variable that did a better job of predicting churn.

## Feature Engineering
---
In an effort to minimize the stress on our machine learning models I created a function that performed statistical analysis on each column based on the data type of the values in that column to determine which, if any, columns were not statistically important and therfore could be dropped. 

I found that phone service had no statistical impact on churn and could be dropped.

I also found that total charges were drastically lower for customers who churned even though their monthly charges were slightly higher on average. This is because most churn happened early in a customers tenure. So, I went ahead and dropped this column as well. 

## Modeling
---
Here we determine the best model to use for predicting churn. I optimized the models for accuracy because the project specifically called for the most accurate model. 

The Random Forest model performed the best with an accuracy of 80% on the train data, 78% on the validate data, and 79% on the test data. Which means that it can be expected to perform with accuracy in the high 70's on any future data. 

## Delivery
---
Here we will complete the goal of the project by delivering actionable suggestions to reduce monthly churn based on our identification of contributing factors. 

Since the project stipulates that the Month-to-Month contract type is not going anywhere, my first suggestion is to offer a slight discount for customers who utilize on of the auotpay options (bank transfer/Credit Card). This address both the fact that people who churn pay more per month on average, and the fact that people who pay by electronic check are more likely to churn than any of the other options combined. 

my second suggestion is to send an automatic offer one month of an additional discount for filling out a survey that is automatically sent to any user that the model predicts will churn. This will likely entice people to stay for at least one month longer, and will ultimately generate even more data that can be used to create more accurate models. 
