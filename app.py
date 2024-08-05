import pandas as pd
import sqlite3
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, swag_from, LazyJSONEncoder, LazyString

import pickle
import re
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model, save_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import CountVectorizer

class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

app = CustomFlaskAppWithEncoder(__name__)

swagger_template = dict(
    info = {
        'title' : LazyString(lambda: "API Documentation for Deep Learning") ,
        'version' : LazyString(lambda: "1.0.0" ),
        'description' : LazyString(lambda: "Sistem API ini digunakan untuk Deep Learning pada Platinum Challenge") 
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "docs",
            "route": "/docs.json"
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

max_features = 10000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
with open("resources_of_lstm/tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle) # Problemnya disini, tokenizer.pickle tidak dipanggil

sentiment = ['negative', 'neutral', 'positive']
print("Senti:", sentiment)

def cleansing (sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9\s]', ' ', string)
    string = re.sub(r'@\w+', '', string)
    string = re.sub(r'\b(rt|RT)\b', '', string) 
    string = re.sub(r'\W+', ' ', string) 
    string = re.sub(r'\s+', ' ', string).strip()  
    string = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(https?://[^\s]+))', ' ', string)
    string = re.sub(' +', ' ', string)
    return string

# Pemanggilan model dan feature Neural Network
count_vet = CountVectorizer()
count_vet = pickle.load(open("resources_of_nn/feature.p", 'rb'))
loaded_model = pickle.load(open("model/model_of_nn/model.p", 'rb'))

# Pemanggilan model dan feature pickle untuk LSTM
file_lstm = open("resources_of_lstm/x_pad_sequences.pickle", 'rb')
feature_file_from_lstm = pickle.load(file_lstm)
file_lstm.close()

model_file_from_lstm = load_model('model/model_of_lstm/model_lstm.h5')

# Endpoint untuk Text Processing menggunakan LSTM
@swag_from("docs/lstm.yaml", methods=['POST'])
@app.route('/lstm', methods=['POST'])

def lstm_text_processing():

    original_text = request.form.get('text')
    print(f"Original text: {original_text}")

    text = [cleansing(original_text)]
    print(f"Cleaned text: {text}")

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    print(f"Tokenized and Padded Sequence: {feature}")

    prediction = model_file_from_lstm.predict(feature)
    print("Prediction:", prediction)

    get_sentiment = sentiment[np.argmax(prediction[0])]
    print("Sentiment:", get_sentiment)

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analyisis using LSTM",
        'data': {
            'text': text, # di lms original_text
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint untuk upload file pada LSTM
@swag_from("docs/lstm_file.yaml", methods=['POST'])
@app.route('/lstm-file', methods=['POST'])

def lstm_file_processing():

    if 'file' not in request.files:
        return jsonify({'status_code': 400, 'description': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status_code': 400, 'description': 'No file selected for uploading'}), 400

    if file:
        filename = secure_filename(file.filename)
        file.save(filename)

    data = pd.read_csv(filename, encoding='latin-1')

    texts = data['Tweet'].to_list()
    sentiment = ['negative', 'neutral', 'positive']
    cleansed_texts = [cleansing(text) for text in texts]

    features = tokenizer.texts_to_sequences(cleansed_texts)
    features = pad_sequences(features, maxlen=model_file_from_lstm.input_shape[1]) # maxlen yang diubah

    predictions = model_file_from_lstm.predict(features)
    sentiments = [sentiment[np.argmax(prediction)] for prediction in predictions]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis File using LSTM",
        'data': [
            {'text': text, 'sentiment': sentiment} for text, sentiment in zip(cleansed_texts, sentiments)
        ]
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint untuk Text Processing menggunakan Neural Network
@swag_from("docs/nn.yaml", methods=['POST'])
@app.route('/nn', methods=['POST'])

def nn_text_processing():

    original_text = request.form.get('text')
    text = count_vet.transform([cleansing(original_text)])

    sentiment = loaded_model.predict(text)[0]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analyisis using Neural Network",
        'data': {
            'text': original_text, ###
            'sentiment': sentiment
        },
    }    
    response_data = jsonify(json_response)
    return response_data

# Endpoint untuk upload file pada Neural Network
@swag_from("docs/nn_file.yaml", methods= ['POST'])
@app.route('/nn-file', methods= ['POST'])

def nn_file_processing():
    
    if 'file' not in request.files:
        return jsonify({'status_code': 400, 'description': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status_code': 400, 'description': 'No file selected for uploading'}), 400

    if file:
        filename = secure_filename(file.filename)
        file.save(filename)

    data = pd.read_csv(filename, encoding='latin-1')

    texts = data['Tweet'].to_list()
    sentiment = ['negative', 'neutral', 'positive']
    cleansed_texts = [cleansing(text) for text in texts]

    features = tokenizer.texts_to_sequences(cleansed_texts)
    maxlen = 10000
    features = pad_sequences(features, maxlen=maxlen) # maxlen yang diubah

    predictions = model_file_from_lstm.predict(features)
    sentiments = [sentiment[np.argmax(prediction)] for prediction in predictions]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis File using Neural Network",
        'data': [
            {'text': text, 'sentiment': sentiment} for text, sentiment in zip(cleansed_texts, sentiments)
        ]
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()