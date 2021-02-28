# Laboratory nr.4 - Model Deployment
#### Table of contents
* [Introduction](#introduction)
* [Routes](#routes)
* [Technologies](#technologies)
* [Setup](#setup)
* [Testing](#testing)
* [Resources](#resources)

## Introduction
The main task of the laboratory work nr.4 is to deploy linear regression model from previous laboratory work.
The client should be able to obtain price prediction of an apartment based on provided characteristics.
This project implements a continuous integration/delivery pipeline, using Heroku and Azure.

## Routes
* *GET* /
* *POST* /predict

## Technologies
Used technologies:
* Python
* Pandas module
* Scikit-learn module
* Pickle module
* Pytest module
* Heroku

## Setup
In order to run the project:
```python app.py```

## Testing
In order to run the tests:
```pytest tests.py```

## Deploy
The application is deployed on the Heroku app: https://utm-fia-lab4.herokuapp.com/

## Resources
UTM Fundamentals of Artificial Intelligence Course
