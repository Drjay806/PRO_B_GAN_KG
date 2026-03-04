FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

COPY . .

# Pre-download artifacts during build so they're available instantly on startup
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='drjay806/PRO-B-GAN-KG-artifacts', local_dir='artifacts', allow_patterns=['*.pt', '*.index', '*.npy', '*.json'])"

EXPOSE 8501

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
