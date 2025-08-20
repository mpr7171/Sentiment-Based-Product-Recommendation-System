# Use official Python base image (Python 3.12 as per your tools, but match your local version)
FROM python:3.8-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (for caching during builds)
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Download NLTK resources (since your model.py downloads them, but ensure in Docker)
RUN python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True);"

# Copy the rest of your project files
COPY . .

# Expose the Flask port (default 5000)
EXPOSE 5000

# Command to run the app (use gunicorn for production; fallback to flask if not installed)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]  # If using gunicorn
CMD ["python", "app.py"] 