![](Images/shopping-1165437_640.jpg)

## Problem Statement
A retail company with multiple outlet stores is having bad revenue returns 
from the stores with most of them facing bankruptcy. This project 
undertakes to review the sales of records from the stores with a view to 
provide useful insights to the company and also to forecast sales outlook for
the next 12-weeks

---

## Project Objective
The Retail Company with multiple outlets across the country are facing 
issues with inventory management. The task is to come up with useful 
insights using the provided data and make prediction models to forecast the 
sales of next twelve weeks

---

## Data Description
The available dataset contains total entries of 6,435*8 having 6,435 rows 
and eight columns. Data Description, various insights from the data.
From the given dataset of the retail company, it is observed that the data 
consists of six thousand four hundred and thirty-five (6,435) records with 
seven features (recorded weekly) as follows:
1. Stores: There are 45 stores and each store has 143 entries of:
2. Date of record (weekly),
3. Total sales record for the week,
4. Holiday flag for the week (1 or 0),
5. Temperature: average temperature recorded during the week
6. Fuel Price: average fuel price for the week,
7. CPI: average Consumer Price Index for the week
8. Unemployment: rate of the unemployment for the week of record

---
## Data Pre-processing Steps and Inspiration
The pre-processing of the data included the following steps:
1. Step 1: Load Data
2. Step 2: Perform ***Exploratory Data Analysis***
    - Confirm number of records in the data and how they are distributed
    - Check data types
    - Check for missing data, invalid entries, duplicates
    - Examine the correlation of the independent features with the target (Weekly_Sales) variables.
    - Check for outliers that are known to distort predictions and forecasts.
3. Step 3: See relations between independent and dependent variables and make inferences.
4. Step 4: Model Predictions, two approaches:
    - Linear Regression Models.
    - Time Series Model (ARIMA, SARIMAX).
5. Step 5: Forecast
6. Step 6: Compare result from different models

---

## Techniques and Data Visualization
![](Images/image1.png)

**Model Approach**:
1. *Regression Models*:
    - Gradient Boosting, Linear Regression and Random forest 
models were also used for the prediction. The best of the three 
predictions will then be compared to the predictions by 
ARIMA or SARIMAX model predictions.
2. *Time Series Model, ARIMA*
    - Using the best ARIMA order, make predictions for the 
selected stores.
    - Forecast using SARIMAX
    - Detrend the dataset if necessary,
    - Using SARIMAX estimate 12 weeks forecast

**Correlation Design**

![](Images/image02.png)

- It was observed from the EDA that the effects of the independent 
variables (Unemployment, Temperature, Holiday_Flag, and CPI) on the 
target variable, weekly sales differ greatly by the store. For example, as 
shown above the effects of unemployment vary by the stores whereas it 
appears to have positive effects on some and negative effects on others. 
The same is also true for Temperature, CPI, and Holiday Flag to some 
extent.
- Premised on the findings, the decision was taken to handle the model, 
predictions by the stores as a single prediction for all the stores may not be 
reasonable given the peculiar conditions prevalent in each region of the 
stores.
- For simplicity and ease of presentation, I have also decide to limit my 
predictions for top eight stores with highest weekly revenue, however the 
model provided cane be used for predictions for other stores or all stores.

---

## Inferences from the Project
### Model Results:
1. ***ARIMA Model*** :
   - **Predictions**: Predictions were performed for eight stores (stores: 20, 4, 14, 13, 2, 
10, 27 and 6) in order of decreasing weekly sales revenue. The 
predictions results are summarized in the Table and graphs below:

![](Images/image03.png)
![](Images/image04.png)

   - **Forecast**: The initial results of the forecast shown above are not very good 
showing evidence of noise which maybe as a result of trends, and the 
observed outliers in the dataset which are distorting the forecasts. As 
a result, the dataset was detrended and the forecast repeated.

![](Images/image05.png)

- The forecast after detrending sales shows the anticipated variabilities. 
However the overall projected sales outlook for the next 12 weeks is 
down for all the stores studied.

![](Images/image06.png)

2. ***Regression Models***:
- The predictions of the three models Gradient Boosting, Linear 
Regression and Random Forest are shown below:

![](Images/image13.png)

### Model Evaluation:
1. ***ARIMA/SARIMAX Models***:
- The model predictions for the selected stores were okay. The 
forecast after detrending was also okay showing variabilities of the 
weekly sales in line with sales history as shown below:

![](Images/image20.png)
![](Images/image07.png)

2. ***Regression Models***:
- The regression model results is summarize
below:

![](Images/image08.png)

- As seen, the mean percentage error is between 3.6% to 9.1% for all 
the models which is within acceptable range. As seen in the prediction 
report table, the results from the three models are very comparable.

![](Images/image28.png)






