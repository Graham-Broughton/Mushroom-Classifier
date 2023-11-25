# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/app

# Copy the current directory contents into the container at /usr/app
COPY . /usr/app

# Install any needed packages specified in Pipfile
RUN pip install pipenv
RUN pipenv install --deploy --ignore-pipfile

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["pipenv", "run", "python", "app.py"]