FROM python:3.7.5-slim-buster

WORKDIR /app

COPY albert-model ./albert-model
COPY requirements.txt ./
RUN apt update && apt install -y build-essential
RUN pip install -U pip
RUN pip install -r ./requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
EXPOSE 80
COPY app.py ./

CMD streamlit run ./app.py --server.port 80