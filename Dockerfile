FROM python:3.9-alpine

RUN apk --no-cache add gcc musl-dev libffi-dev g++ make

WORKDIR /app

COPY requirements.txt .
RUN pip install torch
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "check_new.py"]