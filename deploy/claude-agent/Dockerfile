FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd -m agent && mkdir -p /workspace /data && chown agent /workspace /data
COPY app.py .
USER agent
ENV HOME=/home/agent
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
