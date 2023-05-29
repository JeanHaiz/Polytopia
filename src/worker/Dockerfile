# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM localhost:5000/polyhelper-base:latest

# Set the working directory to /src
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD ./score_visualisation /app/score_visualisation
ADD ./score_recognition /app/score_recognition
ADD ./map_patching /app/map_patching
ADD ./map_analysis /app/map_analysis
ADD ./header_footer_recognition /app/header_footer_recognition

ADD ./database /app/database
ADD ./common /app/common

ADD worker_main.py worker_main.py

# ADD score_visualisation_main.py score_visualisation_main.py
# ADD score_recognition_main.py score_recognition_main.py
# ADD map_patching_main.py map_patching_main.py
# ADD map_analysis_main.py map_analysis_main.py
# ADD turn_recognition_main.py turn_recognition_main.py

# Install any needed packages specified in requirements.txt
# COPY ./worker/requirements.txt requirements.txt

# RUN pip3 install --upgrade pip
# RUN pip3 install -r requirements.txt

# Define environment variable
# ENV NAME World

# Run slash_bot_main.py when the container launches
CMD ["python", "worker_main.py"]