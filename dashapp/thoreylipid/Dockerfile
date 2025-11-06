# Use Python base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy only necessary files
COPY requirements.txt ./
COPY app.py utility.py Final_Combined_Lipidomics.txt ./

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn openpyxl

# Set environment variables
ENV PYTHONPATH=/app
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 --workers=2 --threads=4 --worker-class=gthread --access-logfile -"

# Launch Dash app via Gunicorn
CMD ["gunicorn", "app:server"]
