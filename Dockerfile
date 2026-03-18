FROM python:3.11-slim

WORKDIR /app

# Install system deps for selenium/chrome (scraping) and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip curl gnupg \
    libglib2.0-0 libnss3 libfontconfig1 libgbm1 \
    libwayland-server0 libu2f-udev libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK vader lexicon at build time
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

COPY . .

# Create data and models dirs
RUN mkdir -p data models

EXPOSE 8000

ENV TEMP=/tmp

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
