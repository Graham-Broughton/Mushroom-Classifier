# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/app

# Copy the current directory contents into the container at /usr/app
COPY . /usr/app

# Install any needed packages specified in Pipfile
RUN pip install poetry 
RUN poetry install --no-root

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["poetry", "run", "uvicorn", "--app-dir", ".", "app:app", "--host", "0.0.0.0", "--port", "5000"]