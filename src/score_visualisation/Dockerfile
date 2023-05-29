# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM localhost:5000/polyhelper-base:latest

# Set the working directory to /src
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD ./score_visualisation /app/score_visualisation
ADD ./database /app/database
ADD ./common /app/common
ADD score_visualisation_main.py score_visualisation_main.py

RUN apt-get update
RUN apt-get install gcc ffmpeg libsm6 libxext6 libpq-dev python3.9-dev -y

# Install any needed packages specified in requirements.txt
COPY ./score_visualisation/requirements.txt requirements.txt

# RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Define environment variable
# ENV NAME World

# Run slash_bot_main.py when the container launches
CMD ["python", "score_visualisation_main.py"]