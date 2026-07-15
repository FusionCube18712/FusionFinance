FROM python:3.12-slim

WORKDIR /app

COPY demo/index.html demo/replay.json ./

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=3s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/index.html', timeout=2).read(1)"]

CMD ["python", "-m", "http.server", "8000", "--bind", "0.0.0.0", "--directory", "/app"]
