# inara

# import dependencies

import pandas as pd
import numpy as np
from scipy.stats import f
from scipy import stats
from scipy import linalg


# get training data set

def train(data, p): # takes in a Pandas DataFrame 
    data = data.dropna() # drop rows that contain missing values
    n = len(data.index) # count the number of rows in the data set
    N = int(round((n * p), 0)) # converts p into an integer
    data = data.head(N) # returns the first N rows of the data set
    data = pd.DataFrame(data) # converts data into a DataFrame
    return data # returns the training data set



# get the test data set

def test(data, p): # takes in the same inputs as "train"
    data = data.dropna() # drop rows that contain missing values
    n = len(data.index) # counts the number of rows on the data set
    N = int(round(((1 - n) * p),0)) # converts  p into an integer
    data = data.tail(N) # returns last N rows of data set
    data  = data.reset_index(drop = True) # reset index for "test" data set
    return data # returns the testing data set


# building the multiple linear regression

def reg(x, o): # takes in the training data set and the index of the y variable
    y = x # coppying "data" to use it to create the y_var 
    x = x.drop(x.columns[o], axis = 1) # drops the y variable from the x variables
    x['1'] = 1 # adds a column of 1s to the x variables
    y = y.iloc[ :, o] # creating the y_var
    xt = np.transpose(x) # transposing x
    xtx = np.dot(xt, x) # x transpose dot x
    xty = np.dot(xt, y) # x transpose dot y
    betas = np.linalg.solve(xtx, xty) #----->
    #invxtx = np.linalg.inv(xtx) # inverse of x transpose x
    #betas = np.dot(invxtx, xty) # finding the betas
    return betas 

# build the prediction

def reg_pred(x, y, o): # takes in the test data, the model and the index of the y variable
    z = x # coppying "data" to use it to create the y_var 
    x = x.drop(x.columns[o], axis = 1) # drops the y variable from the x variables
    x['1'] = 1 # adds a column of 1s to the x variables
    xy = np.round(np.dot(x,y),2) # generating the predicted values
    y_value =z.iloc[ :, o] # creating the y_var
    d = {'Predicted':xy, 'Unseen': y_value} # creating a disctionary
    df = pd.DataFrame(data = d) # DataFrame with Predicted vs Unseen data
    return df


# Regression stats

def reg_stats(x, o): # takes in the training data and index of the y variable
    n = len(x.index) # count the number of rows in the dsata set
    n_columns = len(x.columns) # counts the number of columns in the data set
    y = x # coppying "data" to use it to create the y_var
    x = x.drop(x.columns[o], axis = 1) # drops the y variable from the x variables
    x['(Intercept)'] = 1 # adds a column of 1s to the x variables
    y = y.iloc[ :, o] # creating the y_var
    xt = np.transpose(x) # transposing x
    xtx = np.dot(xt, x) # x transpose dot x
    xty = np.dot(xt, y) # x transpose dot y
    betas = np.linalg.solve(xtx, xty) #------>
    ##################
    ##                     ## 
    ##################
    ##                     ##
    ##################
    yt = np.transpose(y) # transpose y
    yty = np.dot(yt, y) # y transpose y dot y
    ybar = np.mean(y) # calulate the average of y
    betat = np.transpose(betas) # transpose of betas
    ybar_sq = np.square(ybar) # square of ybar
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    sst = round(yty - (n * ybar_sq), 7) # sum of the squared total
    sse = round(yty - np.dot(betat, xty), 7) # sum of squared errors
    ssr = round(np.dot(betat, xty) - (n * ybar_sq), 7) # sum of squared regression
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    k = n_columns - 1 # model degrees of freedom
    d3 = n - 1 # total degrees of freedom
    d2 = d3 - k # error degrees of freedom
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    msr = round(ssr / k, 7) # mean squared regression
    mse = round(sse / d2, 7) # mean squared error
    Fstat = round(msr / mse , 7) # f statistic
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    r_sq = round(ssr / sst, 7) # r squared
    adj_r = round((1 - ((1 - r_sq) * (n - 1)) / (n - k - 1)), 7) # adjusted r squared
    se  = round(np.sqrt(1 - adj_r) * np.std(y, ddof = 1), 7) # standard error
    pvalue_f = round(1 - f.cdf(Fstat, k, d2),7)
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    # calculating the standard error of the x variables
    s = np.sqrt(sse / (n - (k + 1))) # variance estimate
    std_error = s * (np.sqrt(np.diagonal(linalg.pinv(xtx)))) # standard error calculation

    # calculating the t - stat
    t_stat = betas / std_error

    # calculate the p - val
    #pval = 1 - stats.t.pdf(np.abs(t_stat), n-1)
    #pval = 1 - stats.t.cdf(np.abs(t_stat), n - (k-1))*2
    pval = stats.t.sf(np.abs(t_stat), n - (k + 1))*2

    
    variables = x.columns # assign the column names to a variable
    dictionary = {"Variables":variables,'Estimate':betas, 'Std Error': std_error, 't value': t_stat, 'Pr(>|t|)': pval}  
    table = pd.DataFrame(data = dictionary)
    table = table.sort_values(by = ['Pr(>|t|)'], ascending = False)
    print('R-squared:' , r_sq,',', ' Adjusted R-squared:', adj_r)
    print('Residual standard error:', se,' with',n, 'observations')
    print('F-statistic:',Fstat, 'on ', k, '&', d2, 'DF',',', 'p-value:', pvalue_f)
    print(1 * "\n")
    print('Coefficients:')
    print(table)


###############################################################################################################
################################    NEW MODEL #### NEW MODEL #### NEW MODEL ###################################
###############################################################################################################

def ltr(y): # takes in the training data set and the index of the y variable
    n = len(y.index) # counting number of rows
    x = list(range(1, n +1)) # generating numbers from 1 to n + 1
    # creating the x variable
    d = {'t': x} 
    x = pd.DataFrame(data = d) 
    x['1'] = 1 # adds a column of 1s to the x variables
    xt = np.transpose(x) # transposing x
    xtx = np.dot(xt, x) # x transpose dot x
    xty = np.dot(xt, y) # x transpose dot y
    betas = np.linalg.solve(xtx, xty) #----->
    return betas


# build the prediction

def ltr_pred(x, y): # takes in the test data and the model
    y_value = x # copying the data for comperison
    n = len(x.index) # counting number of rows
    x = list(range(1, n +1)) # generating numbers from 1 to n + 1
    # creating the x variable
    d = {'t': x} 
    x = pd.DataFrame(data = d) 
    x['1'] = 1 # adds a column of 1s to the x variables
    xy = np.round(np.dot(x,y),2) # generating the predicted values
    d = {'Predicted':xy, 'Unseen': y_value} # creating a disctionary
    df = pd.DataFrame(data = d) # DataFrame with Predicted vs Unseen data
    return df


# Regression stats


def ltr_stats(x): # takes in the training data and index of the y variable
    n = len(x.index) # count the number of rows in the dsata set
    #n_columns = len(x.columns) # counts the number of columns in the data set
    y = x # coppying "data" to use it to create the y_var
    # creating the x variable
    x = list(range(1, n +1)) # generating numbers from 1 to n + 1
    d = {'t': x} 
    x = pd.DataFrame(data = d)
    x['(Intercept)'] = 1 # adds a column of 1s to the x variables
    y = y.iloc[ :, 0] # creating the y_var
    xt = np.transpose(x) # transposing x
    xtx = np.dot(xt, x) # x transpose dot x
    xty = np.dot(xt, y) # x transpose dot y
    betas = np.linalg.solve(xtx, xty) #------>
    ##################
    ##                     ## 
    ##################
    ##                     ##
    ##################
    yt = np.transpose(y) # transpose y
    yty = np.dot(yt, y) # y transpose y dot y
    ybar = np.mean(y) # calulate the average of y
    betat = np.transpose(betas) # transpose of betas
    ybar_sq = np.square(ybar) # square of ybar
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    sst = round(yty - (n * ybar_sq), 7) # sum of the squared total
    sse = round(yty - np.dot(betat, xty), 7) # sum of squared errors
    ssr = round(np.dot(betat, xty) - (n * ybar_sq), 7) # sum of squared regression
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    k = 1 # model degrees of freedom
    d3 = n - 1 # total degrees of freedom
    d2 = d3 - k # error degrees of freedom
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    msr = round(ssr / k, 7) # mean squared regression
    mse = round(sse / d2, 7) # mean squared error
    Fstat = round(msr / mse , 7) # f statistic
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    r_sq = round(ssr / sst, 7) # r squared
    adj_r = round((1 - ((1 - r_sq) * (n - 1)) / (n - k - 1)), 7) # adjusted r squared
    se  = round(np.sqrt(1 - adj_r) * np.std(y, ddof = 1), 7) # standard error
    pvalue_f = round(1 - f.cdf(Fstat, k, d2),7)
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    # calculating the standard error of the x variables
    s = np.sqrt(sse / (n - (k + 1))) # variance estimate
    std_error = s * (np.sqrt(np.diagonal(linalg.pinv(xtx)))) # standard error calculation

    # calculating the t - stat
    t_stat = betas / std_error

    # calculate the p - val
    #pval = 1 - stats.t.pdf(np.abs(t_stat), n-1)
    #pval = 1 - stats.t.cdf(np.abs(t_stat), n - (k-1))*2
    pval = stats.t.sf(np.abs(t_stat), n - (k + 1))*2

    
    variables = x.columns # assign the column names to a variable
    dictionary = {"Variables":variables,'Estimate':betas, 'Std Error': std_error, 't value': t_stat, 'Pr(>|t|)': pval}  
    table = pd.DataFrame(data = dictionary)
    table = table.sort_values(by = ['Pr(>|t|)'], ascending = False)
    print('R-squared:' , r_sq,',', ' Adjusted R-squared:', adj_r)
    print('Residual standard error:', se,' with',n, 'observations')
    print('F-statistic:',Fstat, 'on ', k, '&', d2, 'DF',',', 'p-value:', pvalue_f)
    print(1 * "\n")
    print('Coefficients:')
    print(table)
    
###############################################################################################################
################################    NEW MODEL #### NEW MODEL #### NEW MODEL ###################################
###############################################################################################################

def qtr(y): # takes in the training data set and the index of the y variable
    n = len(y.index) # counting number of rows
    x = list(range(1, n +1)) # generating numbers from 1 to n + 1
    # creating the x variables
    d = {'t': x} 
    x = pd.DataFrame(data = d)
    x['t_sq'] = x['t'] * x['t'] # creating the t_sq variable
    x['1'] = 1 # adds a column of 1s to the x variables
    xt = np.transpose(x) # transposing x
    xtx = np.dot(xt, x) # x transpose dot x
    xty = np.dot(xt, y) # x transpose dot y
    betas = np.linalg.solve(xtx, xty) #----->
    return betas

# build the prediction

def qtr_pred(x, y): # takes in the test data and the model
    y_value = x # copying the data for comperison
    n = len(x.index) # counting number of rows
    x = list(range(1, n +1)) # generating numbers from 1 to n + 1
    # creating the x variable
    d = {'t': x} 
    x = pd.DataFrame(data = d)
    x['t_sq'] = x['t'] * x['t'] # creating the t_sq variable
    x['1'] = 1 # adds a column of 1s to the x variables
    xy = np.round(np.dot(x,y),2) # generating the predicted values
    d = {'Predicted':xy, 'Unseen': y_value} # creating a disctionary
    df = pd.DataFrame(data = d) # DataFrame with Predicted vs Unseen data
    return df

# Regression stats


def qtr_stats(x): # takes in the training data and index of the y variable
    n = len(x.index) # count the number of rows in the dsata set
    #n_columns = len(x.columns) # counts the number of columns in the data set
    y = x # coppying "data" to use it to create the y_var
    # creating the x variable
    x = list(range(1, n +1)) # generating numbers from 1 to n + 1
    d = {'t': x} 
    x = pd.DataFrame(data = d)
    x['t_sq'] = x['t'] * x['t'] # creating the t_sq variable
    x['(Intercept)'] = 1 # adds a column of 1s to the x variables
    y = y.iloc[ :, 0] # creating the y_var
    xt = np.transpose(x) # transposing x
    xtx = np.dot(xt, x) # x transpose dot x
    xty = np.dot(xt, y) # x transpose dot y
    betas = np.linalg.solve(xtx, xty) #------>
    ##################
    ##                     ## 
    ##################
    ##                     ##
    ##################
    yt = np.transpose(y) # transpose y
    yty = np.dot(yt, y) # y transpose y dot y
    ybar = np.mean(y) # calulate the average of y
    betat = np.transpose(betas) # transpose of betas
    ybar_sq = np.square(ybar) # square of ybar
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    sst = round(yty - (n * ybar_sq), 7) # sum of the squared total
    sse = round(yty - np.dot(betat, xty), 7) # sum of squared errors
    ssr = round(np.dot(betat, xty) - (n * ybar_sq), 7) # sum of squared regression
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    k = 2 # model degrees of freedom
    d3 = n - 1 # total degrees of freedom
    d2 = d3 - k # error degrees of freedom
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    msr = round(ssr / k, 7) # mean squared regression
    mse = round(sse / d2, 7) # mean squared error
    Fstat = round(msr / mse , 7) # f statistic
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    r_sq = round(ssr / sst, 7) # r squared
    adj_r = round((1 - ((1 - r_sq) * (n - 1)) / (n - k - 1)), 7) # adjusted r squared
    se  = round(np.sqrt(1 - adj_r) * np.std(y, ddof = 1), 7) # standard error
    pvalue_f = round(1 - f.cdf(Fstat, k, d2),7)
    ##################
    ##                     ##
    ##################
    ##                     ##
    ##################
    # calculating the standard error of the x variables
    s = np.sqrt(sse / (n - (k + 1))) # variance estimate
    std_error = s * (np.sqrt(np.diagonal(linalg.pinv(xtx)))) # standard error calculation

    # calculating the t - stat
    t_stat = betas / std_error

    # calculate the p - val
    #pval = 1 - stats.t.pdf(np.abs(t_stat), n-1)
    #pval = 1 - stats.t.cdf(np.abs(t_stat), n - (k-1))*2
    pval = stats.t.sf(np.abs(t_stat), n - (k + 1))*2

    
    variables = x.columns # assign the column names to a variable
    dictionary = {"Variables":variables,'Estimate':betas, 'Std Error': std_error, 't value': t_stat, 'Pr(>|t|)': pval}  
    table = pd.DataFrame(data = dictionary)
    table = table.sort_values(by = ['Pr(>|t|)'], ascending = False)
    print('R-squared:' , r_sq,',', ' Adjusted R-squared:', adj_r)
    print('Residual standard error:', se,' with',n, 'observations')
    print('F-statistic:',Fstat, 'on ', k, '&', d2, 'DF',',', 'p-value:', pvalue_f)
    print(1 * "\n")
    print('Coefficients:')
    print(table)
    
###############################################################################################################
################################    NEW MODEL #### NEW MODEL #### NEW MODEL ###################################
###############################################################################################################

def hw(y, s): # takes in the data frame and the period
    # extracting the 1st cycle
    S0 = y.head(s)
    S0 = pd.DataFrame(S0)
    mean_S0 = y['Sales'].mean() # average of first cycle
    S0['mean_S0'] = mean_S0 # add the mean to the df
    S0['S0'] = S0['Sales'] / S0['mean_S0']
    return S0




def exp(y, s): # takes in the y variable and the period i.e. weeks, months, days, quarter etc.
    T_t = 0 # initialisingthe trend
    

     
    

    
    
    
    
    


