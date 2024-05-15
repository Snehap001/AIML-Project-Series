import nltk
from nltk.stem import WordNetLemmatizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
lemmatizer = WordNetLemmatizer()
import random
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import json
import spacy
from summary import *
# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")
prev_responses=[]
# Function to extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

# Function to store user information
def store_user_info(user_id, entities):
    if user_id not in user_memory:
        user_memory[user_id] = {}
    user_memory[user_id].update(entities)

# Load model and data
model = load_model('chatbot_model.keras')
intents = json.load(open('intents.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Memory dictionary to store user-specific information
user_memory = {}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json, user_id):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            # Check if the response needs to be personalized
            if '{' in i['responses'][0] and '}' in i['responses'][0]:
                result = personalize_response(random.choice(i['responses']), user_id)
            else:
                result = random.choice(i['responses'])
            break
    return result

def personalize_response(response, user_id):
    if user_id in user_memory:
        user_info = user_memory[user_id]
        for key in user_info:
            placeholder = '{' + key + '}'
            print("placeholder is", placeholder)
            if placeholder in response:

                response = response.replace(placeholder, user_info[key])
    return response

def chatbot_response(text, user_id):
    ints = predict_class(text, model)
    res = get_response(ints, intents, user_id)
    return res

print("Bot is running!")

# Interaction
if __name__ == "__main__":
    print("Start chatting with the bot (type 'quit' to stop)!")
    user_id = "default_user"
    while True:
        message = input("")
        if message.lower() == "quit":
            break

        # Extract entities from the user input
        entities = extract_entities(message)
        if entities:
            store_user_info(user_id, entities)
        
        sum_response=summary(prev_responses)
        
        response = chatbot_response(message, user_id)
        
        prev_responses.append(response)
        print(response)
