FROM python:3.7
COPY . /thermoAI
WORKDIR /thermoAI
RUN pip install -r requirements.txt
