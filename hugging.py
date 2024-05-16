import json
import torch
import random
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Conversation
import nltk
from nltk.stem import WordNetLemmatizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
lemmatizer = WordNetLemmatizer()
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import json
# Load the intents file
with open('intents.json') as f:
    intents = json.load(f)

# Load the model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model2 = load_model('chatbot_model.keras')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
# Initialize the conversational pipeline
chatbot = transformers.pipeline('conversational', model=model, tokenizer=tokenizer)

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
    ERROR_THRESHOLD = 0.65
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
# Function to find the best matching intent for a given user input
# def find_intent(user_input):
#     for intent in intents['intents']:
#         for pattern in intent['patterns']:
#             if pattern.lower() in user_input.lower():
#                 return intent['responses']
#     return None

# Function to handle conversation with memory
def chat_with_memory():
    conversation_history = []
    
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Check if the user input matches any custom intent
        ints = predict_class(user_input, model2)
        if(len(ints)==0):
            custom_response=False
        else:
            tag = ints[0]['intent']
            custom_response=False
            for i in intents['intents']:
                if i['tag']==tag:
                    custom_response=True
                    intent_responses = random.choice(i['responses'])
                    break
        if custom_response:

            print("custom response")
            print("Bot:", intent_responses)
            # conversation_history.append(f"{user_input}")
            # conversation_history.append(f" {response}")
            
        else:
        # Use the model for generating responses
        # conversation_history.append(f"You: {user_input}")
            print("auto response")
            conversation = transformers.Conversation(user_input)
            response = chatbot(conversation,pad_token_id=50256)
            res = str(response)
            bot_response = res[res.find("bot >> ")+6:].strip()
            
            print("Bot:", bot_response)
        

if __name__ == "__main__":
    chat_with_memory()
