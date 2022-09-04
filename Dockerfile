FROM python:3.9.13-buster

COPY src/deployment .

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install markupsafe==2.0.1

EXPOSE 80

CMD ["./app.py", "--host", "0.0.0.0", "--port", "80"]
ENTRYPOINT ["python"]