FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port (optional for Railway)
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
