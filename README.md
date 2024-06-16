# AIML-Project-Series
### Clone the repository
```python
git clone https://github.com/Snehap001/AIML-Project-Series.git
```
## Project 1
### Basic Implementaion of a Chatbot
**Key Features**
* Model trained on an intents.json file using keras to generate customised responses
* Generation of follow-up questions in specific cases using gp2-medium model from transformers
* Implemention of chat-memory for better conversational features
* Personalised responses by storing user informations
### Move to project folder
```python
cd Project1
```
### Installation
* Install the required files
```python
pip install requirements.txt
```

### How to run?
* Train the chatbot on intents.json file
```python
python3 train_chatbot.py
```
* Run the main file
```python
python3 main.py
```


## Project 2
### A complete functional chatbot - Guide for admission related queries
**Key Features**
* intents.json file - including most of the admission related queries
* Model trained on an intents.json file using keras to generate customised responses
* Generation of follow-up questions in specific cases (where the model cannot generate answers) using gp2-medium model from transformers for personalised experience
* Implemention of chat-memory for better conversational features
* Personalised responses by storing user informations
* Web application built using flask
* Backend functionality for extracting data from the college website for information not present in intents.json file.
* jquery used for handling user inputs and chatbot outputs for the web application

### Move to project folder
```python
cd Project2
```
### Installation
* Install the required files
```python
pip install requirements.txt
```
### How to run?
* Train the chatbot on intents.json file

```python
python3 train_chatbot.py
```
* Run the application
```python
python3 app.py
```

## Disease web app

**Data Sources and Model Training**
* Relevant datasets for each disease were gathered from Kaggle.
* Each dataset contained different features, resulting in varied accuracies for the trained models.
* For each disease, six classification models were trained: Logistic Regression, k-NN, kernel SVM, Decision Tree, SVM, and Random Forest.
* After the training process, a feature selection technique was applied to remove features that did not contribute to improving accuracy.
* The Chi-square test was used for feature selection as the datasets included both categorical and numerical features.
* Model performance was evaluated using the cross-validation technique, specifically k-fold cross-validation.
* After selecting the best training models for each disease, the hyperparameters were fine-tuned to enhance accuracy.

**User Interface**
* The web app was built using HTML, CSS, and Bootstrap for the front end, and Flask for the back end.
* The trained models were deployed and are used to provide predictions to users based on their input.

**How to run**
* The folder Disease_web_app contains all the necessary files related to the project.
* The folder model_training_process includes the training process involved in selecting the best models for each disease.
  
### Move to project folder
```python
cd Disease_web_app
```
### Run the web-app
```python
python3 app.py
```
### Run the trained models 
open Jupyter notebook and run the required .ipynb file
```python
cd trained_models
jupyter notebook
```
