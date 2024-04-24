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

@app.route('/train', methods=['GET'])
def run_main():
    try:
        # Запускаем main.py с помощью subprocess.run
        result = subprocess.run(['python', 'main.py'], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Если скрипт выполнен успешно, отправляем уведомление
            rest_url = os.getenv('PY_REST_URL')
            notify_url = 'https://{rest_url}/train'
            notify_response = requests.post(notify_url, json={'status': 'completed', 'output': result.stdout})

            # Проверяем статус ответа от сервера уведомлений
            if notify_response.status_code == 200:
                return jsonify({'status': 'success', 'output': result.stdout, 'notify': 'Notification sent successfully'}), 200
            else:
                return jsonify({'status': 'success', 'output': result.stdout, 'notify': 'Failed to send notification'}), 200
        else:
            return jsonify({'status': 'error', 'output': result.stderr}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PY_PORT'))