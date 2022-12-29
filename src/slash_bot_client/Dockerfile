# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM localhost:5000/polyhelper-base:latest

# Set the working directory to /src
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD ./slash_bot_client /app/slash_bot_client
ADD ./common /app/common
ADD ./database /app/database
ADD slash_bot_main.py slash_bot_main.py


RUN apt-get update
RUN apt-get install gcc ffmpeg libsm6 libxext6 libpq-dev python3.9-dev -y

# RUN apt-get install -y python3-opencv libgl1-mesa-glx python3-lxml libxml2-dev libxslt-dev python-dev libleptonica-dev
# RUN apt-get install -y tesseract-ocr
# RUN apt-get install -y libtesseract-dev

# Install any needed packages specified in requirements.txt
COPY ./slash_bot_client/requirements.txt requirements.txt

# RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Define environment variable
# ENV NAME World

# Run slash_bot_main.py when the container launches
CMD ["python", "slash_bot_main.py"]