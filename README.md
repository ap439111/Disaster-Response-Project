# Disaster-Response-Project

## Table of Contents

1. [Project Motivation](#project_motivation)
2. [Installation](#installation)
3. [Run](#run)
4. [File Descriptions](#file_descriptions)
5. [Results](#results) 


<a name="project_motivation"></a>
## Project Motivation

This project analyses disaster data from [Figure Eight](https://appen.com/) to build a model for an API that calssifies disaster messages. It is a part of a Data Science Nanodegree program by Udacity. The data set contains real messages that were sent during disaster events. The project aims to create a Natural Language Processing(NLP) machine learning pipleine to categorize these events so that the messages can be sent to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classsification results in several categories. The web app will also display visualizations of the data.

The project is has three major categories:

  1. Build an ETL pipeline; extract data from sources, clean the data and save them in a SQLite database.
  2. Build NLP machine learning (ML) pipleine to classify messages in several categories.
  3. Build a web app which can calssify the message using models in the real time.
  
<a name="installation"></a>
## Installation  

  Requires Python 3+
  
  The repository can be cloned by: 
  
    git clone  https://github.com/ap439111/Disaster-Response-Project.git
    

<a name="run"></a>
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
     
<a name="files_descriptions"></a>
## Files Descriptions

  There are three folders:
  
   1. data
        
        This folder contains following files:
        
          i.  ETL_Pipeline_Preparation.ipynb: ETL Pipeline preparation notebook
          ii. process_data.py: ETL pipleine (python script) to extract data and features, transform the data and load it in SQLite database
          iii. Data files: messages.csv and categories.csv
          iv. disaster_response.db: Cleanded SQLite database file
              
  2. model
  
        This folder contains following files:
        
        
          i.  ML_Pipeline_Preparation.ipynb: ML pipeline preparation notebook
          ii. train_classifier.py: ML pipeline to load the SQLite database, train the ML model and save the model as pickle file. 
          iii. classifier.pkl: Saved model
          
  3. app
  
        This folder contains following folder and files:
        i. 




