# syntax=docker/dockerfile:1
FROM python:3.9.13-buster

RUN mkdir app
ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install markupsafe==2.0.1

EXPOSE 8181

CMD ["gunicorn", "deploy.app:app", "--bind=0.0.0.0:8181"]