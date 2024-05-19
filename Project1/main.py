import json
import spacy
import torch
import random
import transformers
from question_generation import ConversationContext
from follow_up_ques import generate_followup_questions
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Conversation
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.stem import WordNetLemmatizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import json

with open('intents.json') as f:
    intents = json.load(f)


model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model2 = load_model('chatbot_model.keras')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

chatbot = transformers.pipeline('conversational', model=model, tokenizer=tokenizer)
context_manager = ConversationContext()
user_memory = {}
user_id = "default_user"

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
    ERROR_THRESHOLD = 0.50
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def is_question(sentence):
    doc = nlp(sentence)

    if sentence.strip().endswith('?'):
        return True
    interrogatives = {'what', 'where', 'when', 'which', 'who', 'whom', 'whose', 'why', 'how'}
    auxiliaries = {'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'do', 'does', 'did'}
    for token in doc[:2]: 
        if token.lemma_.lower() in interrogatives or token.lemma_.lower() in auxiliaries:
            return True   
    return False

def personalize_response(response, user_id):
    if user_id in user_memory:
        user_info = user_memory[user_id]
        for key in user_info:
            placeholder = '{' + key + '}'
          
            if placeholder in response:

                response = response.replace(placeholder, user_info[key])
            
    if '{' in response and '}' in response:
        prefix=""
        i=0
        while response[i]!='{':
            prefix+=response[i]
            i+=1
        med=""
        i+=1
        while response[i]!='}':
            med+=response[i]
            i+=1
        i+=1
        suffix=response[i:]
        ans=prefix+suffix
        return ans
    return response

def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities
def store_user_info(user_id, entities):
    if user_id not in user_memory:
        user_memory[user_id] = {}
    user_memory[user_id].update(entities)
def chat_with_memory():
  
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        print("You: ",end="")
        user_input = input()
        if user_input.lower() == 'quit':
            break
        
        new_ques=user_input
        isQues=is_question(user_input)
        if(isQues):
            new_ques=context_manager.combine_questions(user_input)

            context_manager.add_question(user_input)
        else:
            entities = extract_entities(user_input)
            if entities:
                store_user_info(user_id, entities)

        if((not(isQues) and not(entities)) ):
            follow_up=generate_followup_questions(user_input)
            for q in follow_up:
                print("Bot: ",q)
            print("You: ",end="")
            user_input = input()
            new_ques=user_input
        
        ints = predict_class(new_ques, model2)
        intent_responses=""
        if(len(ints)==0):
            custom_response=False
        else:
            tag = ints[0]['intent']
            custom_response=False
            for i in intents['intents']:
                if i['tag']==tag:
                    custom_response=True
                    if '{' in i['responses'][0] and '}' in i['responses'][0]:
                        intent_responses = personalize_response(random.choice(i['responses']), user_id)
                    else:
                        intent_responses = random.choice(i['responses'])
                    break
        if custom_response:

            print("Bot: ",intent_responses)

            
        else:
            print("sorry I am unable to infer what You are trying to communicate")
            
        
if __name__ == "__main__":
    chat_with_memory()
