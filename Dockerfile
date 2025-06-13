FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

RUN chmod +x start.sh

# Expose port (optional for Railway)
EXPOSE 8000


CMD ["./start.sh"]
