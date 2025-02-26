FROM python:3.10-slim

# set the working directory in the container
WORKDIR /app

# copy the current directory contents into the container at /app
COPY asr_api.py /app/asr_api.py
COPY requirements.txt /app/requirements.txt

# install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# expose port 8001 to outside world
EXPOSE 8001

# launch fast api server
CMD ["uvicorn", "asr_api:app", "--host", "0.0.0.0", "--port", "8001"]