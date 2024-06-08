from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    # Here, you would add your code to predict the disease based on the symptoms
    # For now, let's just redirect to the homepage and print something on the console
    print("Predict Disease button was clicked.")
    return redirect(url_for('home'))

@app.route('/symptom_analysis', methods=['POST'])
def symptom_analysis():
    # Here, you would add your code for symptom analysis
    # For now, let's just redirect to the homepage and print something on the console
    print("Symptom Analysis button was clicked.")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
