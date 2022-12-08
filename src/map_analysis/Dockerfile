# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /src
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD ./map_analysis /app/map_analysis
ADD ./database /app/database
ADD ./common /app/common
ADD map_analysis_main.py map_analysis_main.py

RUN apt-get update
RUN apt-get install gcc ffmpeg libsm6 libxext6 libpq-dev python3.9-dev -y

# Install any needed packages specified in requirements.txt
COPY ./map_analysis/requirements.txt requirements.txt

# RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Define environment variable
# ENV NAME World

# Run slash_bot_main.py when the container launches
CMD ["python", "map_analysis_main.py"]