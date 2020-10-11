# Disaster-Response-Project

## Table of Contents

1. [Project Motivation](#Project-Motivation)
2. [Installation](#Installations)
3. [Run](#Run)
4. [File Descriptions](#File-Descriptions)
5. [Results](#Results) 

## Project Motivation

This project analyses disaster data from [Figure Eight](https://appen.com/) to build a model for an API that calssifies disaster messages. It is a part of a Data Science Nanodegree program by Udacity. The data set contains real messages that were sent during disaster events. The project aims to create a Natural Language Processing(NLP) machine learning pipleine to categorize these events so that the messages can be sent to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classsification results in several categories. The web app will also display visualizations of the data.

The project is has three major categories:

  1. Build an ETL pipeline; extract data from sources, clean the data and save them in a SQLite database.
  2. Build NLP machine learning (ML) pipleine to classify messages in several categories.
  3. Build a web app which can calssify the message using models in the real time.
  
## Installation  

  Requires Python 3+
  
  The repository can be cloned by: 
  
    git clone  https://github.com/ap439111/Disaster-Response-Project.git
    
  
## Run

     > cd Disaster-Response-Project
     
  To run the ETL pipeline:
  
      > python data/process_data.py data/messages.csv data/categories.csv data/disaster_response.db
      
  To run the ML pipeline:
  
      > python models/train_classifier.py data/disaster_response.db models/classifier.pkl
      
  To run the web app:
  
      > cd app
      > python run.py
      
  To display app, go to:
  
       http://0.0.0.0:3001/
     
  
  




