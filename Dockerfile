# Use an official Python runtime as a parent image
FROM python:3.7-slim 
# Set the working directory to /notebooks 
WORKDIR /notebooks
# install necessary build packages 
RUN  apt-get update && apt-get install -y gcc libsndfile1-dev libsm6 libxrender1 libxext-dev 
# Copy the current directory contents into the container at /app
COPY requirements.txt /notebooks
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org --no-cache -r requirements.txt
# Make port 8888 available to the world outside this container 
EXPOSE 8888 
