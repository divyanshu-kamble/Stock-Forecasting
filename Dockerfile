FROM python:buster

WORKDIR /app
COPY requirements.txt ./

COPY application.py ./

RUN apt-get update && apt-get install -y gcc python3-dev libtbb2

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["application.py"]