FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY classifier /app/classifier
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]