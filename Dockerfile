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

EXPOSE 7860

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
