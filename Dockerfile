FROM python:3.9

RUN apt-get update && apt-get install -y \
    gcc \
    libc6-dev \
    libffi-dev \
    g++ \
    make
    
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt

RUN python -m nltk.downloader stopwords

RUN python -m nltk.downloader wordnet


COPY . .

CMD ["python", "check_new.py"]