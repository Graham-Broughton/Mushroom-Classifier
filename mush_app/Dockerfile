# Use an official Python runtime as a parent image
FROM python:3.11-slim as build

# Set the working directory in the container
WORKDIR /mush_app

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# RUN set -ex \
#     # Create a non-root user
#     && addgroup --system --gid 1001 appgroup \
#     && adduser --system --uid 1001 --gid 1001 --no-create-home appuser \

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in Pipfile
RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
