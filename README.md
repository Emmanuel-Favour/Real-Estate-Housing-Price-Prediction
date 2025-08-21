# Real Estate House Pricing Prediction 🏡 

A machine learning end to end project from model development to Model deployment that predicts house prices based on various features such as square footage, location, number of bedrooms, waterfront view, and more. This project explores data preprocessing, feature engineering, polynomial regression, and regularization (Ridge Regression) to build an optimized predictive model.

### Table of Contents

Overview -----> Dataset -----> Project Workflow -----> Modeling -----> Results -----> Deployment -----> Technologies Used -----> How to Run -----> Future Improvements 

*Author: Favour Emmanuel*

### Overview

The real estate market is highly influenced by several factors, and accurately predicting house prices is crucial for buyers, sellers, and investors.
This project uses supervised machine learning techniques to:

* Explore and visualize housing data

* Identify relationships between features and price

* Train regression models with polynomial feature expansion

* Apply Ridge Regression to reduce overfitting

* Deploy the final model with FastAPI + Uvicorn

### Dataset

*Source: Kaggle Housing Prices Dataset (or specify your actual source)*

* Target Variable: price

Features include:

* sqft_living – Living area square footage

* bedrooms – Number of bedrooms

* bathrooms – Number of bathrooms

* waterfront – Waterfront view (0/1)

* floors, condition, grade

and more…

### Project Workflow

Data Preprocessing -----> Feature scaling -----> Exploratory Data Analysis (EDA) -----> Feature Engineering -----> Model Training -----> Model Evaluation

> Results

> Training R² Score: ~0.85

> Testing R² Score: ~0.82

Ridge Regression successfully reduced overfitting compared to standard polynomial regression.

### Model Deployment

The trained Ridge Regression model was deployed using FastAPI, providing a lightweight and scalable REST API for real estate price prediction. The API accepts structured input features
(e.g., sqft_living, bedrooms, bathrooms, grade, zipcode, etc.) in JSON format and returns predicted housing prices in real time. Swagger UI is integrated for interactive testing and documentation,
making it easy for users to query the model endpoints and validate predictions.
