# syntax=docker/dockerfile:1
FROM python:3.9.13-buster

RUN mkdir app
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install markupsafe==2.0.1

WORKDIR /src/deployment
EXPOSE 5000

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]