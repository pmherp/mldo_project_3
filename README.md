# Udacity Machine Learning DevOps Engineer - Project 3

Predict Deploy a Machine Learning Pipeline with GitHub Actions and Render
Powered by: Udacity

## Introduction
The Goal of this project is to deploy a full ML Pipeline using GitHub Actions for CI and Render for CD. 

The model itself uses census data to predict wether an individual earns more or less than 50K a year. For that, the model uses a RandomForestClassifier with the standard settings as hyperparameters.

## Installation
This project runs on python version 3.8. Make sure you have the right Python version installed on your local machine.

Install the dependencies and libraries with the requirements.txt provided in this repository:

pip install -r requirements.txt

## How to run the Code
To run main.py that holds the entire Data Science Process Code, run the following in your terminal:

python main.py

This will automatically run the entire code on your local machine.

To deploy the code to Render, setup a Webservice on Render, connect your Repository with it and deploy the code using Render.

## Testing
Through GitHub Actions, there are automatic tests run with every push to your repository, when you activate GitHub Actions in your repository.

Tests include testing the API deployment and the model performance.

To run the tests locally, run:

python test_api.py in the "starter/starter/" directory or python test_model.py in the "starter/starter/ml" directory.

## Results
The model returns a classification on wether an individual earns more or less than 50K a year.

License
MIT

I want to give credit for the data and basic code provide by Udacity's Machine Learning DevOps Nanodegree program.
