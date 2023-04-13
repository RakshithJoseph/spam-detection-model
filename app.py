import numpy as np
import spacy
import pandas as pd
from flask import Flask, render_template, request, send_file
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
vect = pickle.load(open('vec.pkl', 'rb'))
model = pickle.load(open('mdel.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict1', methods=['GET', 'POST'])
def predict1():

    comment = request.form.get('msg')
    data = [comment]
    input_data_features = vect.transform(data).toarray()

    my_prediction = model.predict(input_data_features)
    return render_template('home.html', prediction_text='The entered message is {}'.format(my_prediction))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        
       
        if not uploaded_file:
            return render_template('home.html', error_message='Please upload a file')
        
        try:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        except:
            return render_template('home.html', error_message='Invalid file format. Please upload a CSV file')
        
        en = spacy.load('en_core_web_sm')
        sw_spacy = en.Defaults.stop_words
        df['clean'] = df['v2'].apply(lambda x: ' '.join(
            [word for word in x.split() if word.lower() not in (sw_spacy)]))
        
        count = vect.transform(df['clean'])
        
        
        
        predictions = model.predict(count)
        
        df['Prediction'] = predictions
        
        df.to_csv('predictions.csv', index=False)
        
        return send_file('predictions.csv', as_attachment=True)
    
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
