# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /src
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

RUN apt-get autoclean && apt-get update
RUN apt-get install -y python3-opencv ffmpeg libsm6 libxext6 libgl1-mesa-glx python3-lxml libxml2-dev libxslt-dev python-dev libleptonica-dev 
RUN apt-get install -y tesseract-ocr
RUN apt-get install -y libtesseract-dev

# Install any needed packages specified in requirements.txt
COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Define environment variable
# ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]