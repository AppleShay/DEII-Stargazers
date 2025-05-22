FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py .
COPY final_model.pkl .
COPY requirements.txt .
COPY templates/ templates/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port used by Flask
EXPOSE 5100

# Run the Flask app
CMD ["python", "app.py"]
