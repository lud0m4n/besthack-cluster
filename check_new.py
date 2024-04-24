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

load_dotenv()
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
with open('kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

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
    data = request.get_json()
    new_message = data['message']
    embedding = get_message_embedding(new_message)
    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(float)
    cluster_label = kmeans.predict([embedding])[0]
    cluster_center = kmeans.cluster_centers_[cluster_label]
    distance = np.linalg.norm(embedding - cluster_center)
    threshold = 10
    if distance < threshold:
        response = {
            "message": f"Message is classified into cluster {cluster_label} with distance {distance}."
        }
    else:
        response = {
            "message": "Message does not fit into any existing cluster."
        }

    # Отправляем результат на другой сервер
    try:
        rest_url = os.getenv('PY_REST_URL')
        send_url = 'https://{rest_url}/cluster'
        requests.post(send_url, json=response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Возвращаем ошибку, если что-то пошло не так

    return jsonify(response)  # Возвращаем результат также локальному клиенту

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PY_PORT'))