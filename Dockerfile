# syntax=docker/dockerfile:1
FROM python:3.9.13-buster

RUN mkdir app
ADD . /app
WORKDIR /app/deployment

RUN pip install -r requirements.txt
RUN pip install markupsafe==2.0.1

EXPOSE 5000

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]