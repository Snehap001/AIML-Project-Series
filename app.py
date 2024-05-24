from flask import Flask, request, jsonify, flash, render_template, redirect, url_for
import requests
from bs4 import BeautifulSoup
import spacy
import middle

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.static_folder = 'static'
user_memory={}
def extract_keywords(text):
    doc = middle.main.nlp(text)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]


def process_query(query):
    keywords = extract_keywords(query)

    return scrape_website(keywords)

def scrape_website(keywords):
    url = "https://home.iitd.ac.in/faq.php"  
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p') if any(keyword in p.get_text() for keyword in keywords)]
        return ' '.join(paragraphs) if paragraphs else "No relevant information found."
    else:
        return "Failed to retrieve information."
    
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    query = request.args.get('msg')
    response="Error not found"
    if query:
        response=middle.user_response(query,user_memory)
        if response=="No info available":
            print("webpage response")
            response = process_query(query)
            print(response)
            if response=="Failed to retrieve information."or response=="No relevant information found." :
                response="Sorry I am unable to infer what you are trying to ask?"

        response = response.replace('\n', '<br>')
        return response
    
    else:
        response = response.replace('\n', '<br>')
        return  response
  
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
