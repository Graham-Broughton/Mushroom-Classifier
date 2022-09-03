FROM jupyter/base-notebook

COPY src/deployment .

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install markupsafe==2.0.1
