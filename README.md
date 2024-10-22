# Simple Linear Regression for Predicting CO2 Emissions

## Overview
This repository contains a data analysis project focused on applying simple linear regression to predict CO2 emissions based on engine size and fuel consumption. The project explores the relationship between vehicle characteristics and CO2 emissions, providing insights into the effectiveness of engine size and fuel consumption as predictors.

## Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Simple Linear Regression Model](#simple-linear-regression-model)
4. [Results](#results)


## Introduction
The project aims to analyze how engine size and fuel consumption impact CO2 emissions using a simple linear regression model. It demonstrates the application of regression analysis in predicting environmental impacts based on vehicle specifications.

## Dataset
The dataset used (`FuelConsumptionCO2.csv`) includes information on vehicle characteristics and corresponding CO2 emissions:
- `ENGINESIZE`: Engine size of the vehicle.
- `CYLINDERS`: Number of cylinders.
- `FUELCONSUMPTION_COMB`: Combined fuel consumption in liters per 100 kilometers.
- `CO2EMISSIONS`: CO2 emissions in grams per kilometer.

## Simple Linear Regression Model
- Data Preparation: Load and preprocess the dataset.
- Train-Test Split: Divide the data into training and testing sets.
- Model Training: Fit a linear regression model using scikit-learn.
- Model Evaluation: Evaluate the model's performance using metrics like mean absolute error and R-squared score.
- Visualization: Visualize the regression line and actual data points to assess model fit.

## Results
- Model Performance: Achieved an R-squared score of 0.75, indicating a moderate fit of the model to predict CO2 emissions based on engine size.
- Alternative Model: Explored predicting CO2 emissions using fuel consumption, achieving a mean absolute error of 20.36.

## Installation
To run the notebook locally, ensure Python and the following libraries are installed:
```bash
pip install numpy pandas matplotlib scikit-learn
