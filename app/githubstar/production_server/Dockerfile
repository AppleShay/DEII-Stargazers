FROM python:3.10-slim

WORKDIR /app

# Copy all files inside production_server/
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential gcc libglib2.0-0 libsm6 libxext6 libxrender-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV FLASK_APP=app.py
EXPOSE 5100

CMD ["python", "app.py"]
