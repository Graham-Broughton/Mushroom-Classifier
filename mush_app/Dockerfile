# # Use an official Python runtime as a parent image
# FROM python:3.11-slim as build

# # Set the working directory in the container
# WORKDIR /mush_app

# ENV PIP_DEFAULT_TIMEOUT=100 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1 \
#     PIP_NO_CACHE_DIR=1 \
#     PYTHONUNBUFFERED=1 \
#     PORT=8080

# # Copy the current directory contents into the container at /app
# COPY . .

# # Install any needed packages specified in Pipfile
# RUN pip install -r requirements.txt

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

FROM python:3.11-slim as build

ENV PIP_DEFAULT_TIMEOUT=100 \
    # Allow statements and log messages to immediately appear
    PYTHONUNBUFFERED=1 \
    # disable a pip version check to reduce run-time & log-spam
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # cache is useless in docker image, so disable to reduce image size
    PIP_NO_CACHE_DIR=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/mush_app/mush-classifier-main.json

ARG POETRY_VERSION=1.7.1

WORKDIR /mush_app
COPY pyproject.toml poetry.lock ./

RUN pip install "poetry==$POETRY_VERSION" \
    && poetry install --no-root --no-ansi --no-interaction \
    && poetry export -f requirements.txt -o requirements.txt


### Final stage
FROM python:3.11-slim as final

WORKDIR /mush_app

COPY --from=build /mush_app/requirements.txt .

RUN set -ex \
    # Create a non-root user
    && addgroup --system --gid 1001 appgroup \
    && adduser --system --uid 1001 --gid 1001 --no-create-home appuser \
    # Upgrade the package index and install security upgrades
    && apt-get update \
    && apt-get upgrade -y \
    # Install dependencies
    && pip install -r requirements.txt \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

# Set the user to run the application
USER appuser
