# Diamond-Price-Prediction

Using Data Science with Machine Learning to detect the price of diamonds using significant features given by the most linked features that are taken into consideration when evaluating price by diamond sellers.


## Table of Contents
* [Introduction](#introduction)
* [Dataset General info](#dataset-general-info)
* [Technologies](#technologies)
* [Features](#features)
* [Project Checklist](#project-checklist)
* [Result](#result)
* [Sources](#sources)


## Introduction
This project is a part of my training at SHAI FOR AI.
In a world of speed and development, and the great expansion of technology based on artificial intelligence, machine learning and its uses in many scientific and practical fields, academic and professional, as an example professionally in financial forecasts, which we find its importance based on the correct and accurate prediction of the problem and determining the possibility of addressing it and solving it in the most accurate ways, and scientific methods and evaluating it on the best possible standards.\
Based on this introduction, I present to you my project in solving the problem of diamond price prediction, and my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning.\
Hoping to improve it gradually in the coming times.

![1](https://user-images.githubusercontent.com/48035751/194652434-42302240-1d26-4e3d-a6e0-c70b03bbed53.jpeg)


## Dataset General info
**General info about the dataset:**

Context

This is a classic dataset contains the prices and other attributes of almost 54,000 diamonds, but in SHAI competition project we use a subset of the full data which contain only 43040 diamonds.

* Content price price in US dollars (\$326--\$18,823)

* carat weight of the diamond (0.2--5.01)

* cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)

* color diamond colour, from J (worst) to D (best)

* clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

* x length in mm (0--10.74)

* y width in mm (0--58.9)

* z depth in mm (0--31.8)

* depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

* table width of top of diamond relative to widest point (43--95)

![2](https://user-images.githubusercontent.com/48035751/194652738-26f679fe-0bc9-44a4-928b-a6cc4b9867e8.jpeg)

## Technologies
* Programming language: Python.
* Libraries: OS, Numpy, Matplotlib, Pandas, Seaborn, sklearn, xgboost. 
* Application: Colaboratory.


## Features
* I present to you my project solving the problem of diamond price prediction using a lot of effective algorithm and techniques with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it in the best possible ways and the current capabilities using Machine Learning.

### To Do:
Briefly about the process of the project work:
* Take a comprehensive view of the data contained within the data set.
* Structuring problem.
* Choosing a performance measure algorithm: here RMSE algorithm was chosen.
* Hypothesis testing: all of my hypotheses here were in the Machine Learning field.
* Preparing the work environment to deal with the data and solving the problem.
* Download the dataset.
* Do a quick look at the data structure
* Build the test set.
* Explore and display the data to get ideas from it: this stage aims to extract ideas and a deeper understanding of the data and the goal of the problem.
* Searching for correlation: correlation test resulting from merging a set of descriptors with each other, and manipulate with features.
* Data cleaning.
* Dealing with texts and categorical data.
* Build custom transformers and value converters.
* Feature standardization.
* Model selection and training: choosing the optimal model for training data, and at this stage a lot of machine learning algorithms related to Regression were tested and the analysis was based on the optimum ones: Decision Tree, Random Forest, Grandient Boosting, XGBoosting,  SVR, Linear Regression, Lasso, Ridge.
* Training and evaluation on the dataset.
* Better evaluation with cross-validation, and learning curve.
* Get the optimal setting of the model: setting the model best parameters using GridSearchCV, RandomizedSearchCV.
* Analyze the best model and analyze the errors.
* Evaluate the model on the test data.


## Project Checklist
### Part 1 : Data Analysis
  
* Import necessary libraries
* Get the diamond price prediction dataset
* Discover and visualize the data to gain insights (Data preprocessing)
* Check for any ZERO value
* Check for Duplicate rows
* View and remove outliers
* Convert to log scale
* Convert categorical variables to numerical column using labelencoder

### Part 2 :Model Building
  
* Import ML libraries.
* Split the data into test and train.
* Build a pipeline of standard scalar and model for different regressors.
* Fit all the models on training data.
* Get mean of cross-validation on the training set for all the models for Pick the model with the best cross-validation score.
* Fit the best model on the training set.

## Result 

the best Model is RandomForest by score 0.9915925990836592

R^2: 0.9920796250484655

Adjusted R^2: 0.9920607520342839

MAE: 0.05877902410580197

MSE: 0.0068323458915730065

RMSE: 0.08265800561090865



## Sources
This data was taken from SHAI competition (Diamond Price Prediction competition)\
(https://www.kaggle.com/t/0aa9c9c6994548aba1f257a94e1c59cc)
