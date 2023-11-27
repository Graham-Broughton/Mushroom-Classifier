# Use an official Python runtime as a parent image
FROM python:3.10-slim 

# Set the working directory in the container
WORKDIR /mush_app

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in Pipfile
RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
