from flask import Flask, request, jsonify
import requests  # Импортируем библиотеку requests
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from dotenv import load_dotenv
import os
import subprocess
import pandas as pd

load_dotenv()
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(привет|добрый день|доброе утро|добрый вечер|здравствуйте)\b', '', text)
    tokens = word_tokenize(text, language='russian')
    tokens = [token for token in tokens if token not in stopwords.words('russian')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

def get_message_embedding(message):
    message = preprocess_text(message)
    encoded_input = tokenizer(message, padding=True, truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return np.asarray(model_output.pooler_output[0].numpy())

@app.route('/cluster', methods=['POST'])
def classify_message():
    with open('data/kmeans_model.pkl', 'rb') as file:
        kmeans = pickle.load(file)
    
    data = request.get_json()
    new_message = data['message']
    embedding = get_message_embedding(new_message)
    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(float)
    cluster_label = kmeans.predict([embedding])[0]
    cluster_center = kmeans.cluster_centers_[cluster_label]
    distance = np.linalg.norm(embedding - cluster_center)
    threshold = 11
    cluster_data = pd.read_csv('data/clustered_messages.csv')
    cluster_counts = cluster_data['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster', 'count']
    cluster_count = cluster_counts.loc[cluster_counts['cluster'] == cluster_label, 'count'].iloc[0]
    # Ответ сервера
    if distance < threshold:
        response = {
            "cluster_index": int(cluster_label),
            "cluster_frequency": int(cluster_count)
        }
    else:
        response = {
            "cluster_index": 14,
            "cluster_frequency": 0
        }

    return jsonify(response)  # Возвращаем результат также локальному клиенту

@app.route('/train', methods=['GET'])
def train():
    try:
        # Запускаем main.py с помощью subprocess.run
        result = subprocess.run(['python', 'main.py'], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Если скрипт выполнен успешно, возвращаем успешный ответ
            return jsonify({'status': 'success', 'output': result.stdout}), 200
        else:
            # Если скрипт завершился с ошибкой, возвращаем ошибку
            return jsonify({'status': 'error', 'output': result.stderr}), 400
    except Exception as e:
        # В случае возникновения исключения возвращаем сообщение об ошибке
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=os.getenv('PY_HOST'), debug=True, port=os.getenv('PY_PORT'))