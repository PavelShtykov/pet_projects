FROM python:3.8-slim-buster

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY ./src .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# define the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python", "./run.py"]
