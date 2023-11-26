# Use an official Python runtime as a parent image
FROM python:3.10-slim as builder

# Set the working directory in the container
WORKDIR "/app"

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Copy the current directory contents into the container at /app
COPY pyproject.toml poetry.lock ./
COPY ./mush_app mush_app
COPY ./README.md README.md

# Install any needed packages specified in Pipfile
RUN pip install poetry
RUN python -m venv /venv

RUN . /venv/bin/activate && poetry install --no-root --with deploy
RUN . /venv/bin/activate && poetry build

FROM python:3.10-slim as final

WORKDIR "/app"

COPY --from=builder /venv /venv
COPY --from=builder /app/dist .
COPY ./mush_app mush_app

RUN . /venv/bin/activate && pip install *.whl

CMD ["/venv/bin/python", "-m", "uvicorn", "--app-dir", "mush_app", "app:app", "--host", "0.0.0.0", "--port", "8080"]