FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download artifacts during build so they're available instantly on startup
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='drjay806/PRO-B-GAN-KG-artifacts', local_dir='artifacts', allow_patterns=['*.pt', '*.index', '*.npy', '*.json'])"

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
