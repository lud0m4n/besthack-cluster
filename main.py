import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.manifold import TSNE
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import pickle
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Загрузка данных
with open("/data/dataset_hack.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Создание DataFrame
df = pd.DataFrame(data)

# Выборка колонки с сообщениями
messages = df['message'].tolist()
def preprocess_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'\b(привет|добрый день|доброе утро|добрый вечер|здравствуйте)\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))  # Удаление знаков пунктуации
    tokens = word_tokenize(text, language='russian')  # Токенизация
    tokens = [token for token in tokens if token not in stopwords.words('russian')]  # Удаление стоп-слов
    lemmatizer = WordNetLemmatizer()  # Инициализация лемматизатора
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Лемматизация
    return " ".join(tokens)

# Применение функции предобработки к каждому сообщению в списке
messages = [preprocess_text(message) for message in messages]
tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

def embedding(messages):
    embeddings_list = []
    for message in messages:
        encoded_input = tokenizer(message, padding=True, truncation=True, max_length=64, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings_list.append(model_output.pooler_output[0].numpy())
    embeddings = np.asarray(embeddings_list)
    return embeddings


from sklearn.cluster import KMeans

embeddings = embedding(messages)
# Определение количества кластеров (используя функцию determine_k)
k_min = 2
k_max = 25
clusters = range(k_min, k_max + 1)
inertia = [KMeans(n_clusters=k, random_state=42).fit(embeddings).inertia_ for k in clusters]
plt.plot(clusters, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()


elbow_data = pd.DataFrame({
    'Clusters': clusters,
    'Inertia': inertia
})
elbow_filename = '/data/elbow_data.csv'
elbow_data.to_csv(elbow_filename, index=False)
print(f"Elbow data saved to {elbow_filename}.")
#
k = 13
#

# После визуализации выбираем k в зависимости от графика

kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)


cluster_counts = df['cluster'].value_counts()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.title('Frequency of Messages by Cluster')
plt.show()


# Сохранение модели KMeans
kmeans_model_filename = '/data/kmeans_model.pkl'
with open(kmeans_model_filename, 'wb') as file:
    pickle.dump(kmeans, file)

print(f"KMeans model saved to {kmeans_model_filename}.")


tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)
tsne_data = pd.DataFrame(embeddings_2d, columns=['t-SNE Feature 1', 't-SNE Feature 2'])
tsne_data['cluster'] = df['cluster']
tsne_filename = '/data/tsne_data.csv'
tsne_data.to_csv(tsne_filename, index=False)
print(f"t-SNE data saved to {tsne_filename}.")
# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=df['cluster'], cmap='viridis', alpha=0.5)
plt.colorbar()
plt.title('t-SNE Visualization of Message Embeddings')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.show()
# Для вывода примеров сообщений из каждого кластера
for cluster in sorted(df['cluster'].unique()):
    print(f"Cluster {cluster}:")
    print(df[df['cluster'] == cluster]['message'].head())
csv_filename = '/data/clustered_messages.csv'
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}.")

