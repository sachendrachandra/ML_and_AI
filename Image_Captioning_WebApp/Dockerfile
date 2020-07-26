#Use pyhton as base image 
FROM python:3.7-slim

#Use working directory /app
WORKDIR /app

#copy all the content of current directory to /app
ADD . /app

#installing required packages
RUN pip install --trusted-host pypi.org -r requirements.txt

#OPen port 5000
EXPOSE 5001

#Set environment variable
ENV NAME OpentoAll

#Run python program
CMD ["python","app.py"]
