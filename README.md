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
