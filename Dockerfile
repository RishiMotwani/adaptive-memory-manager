FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.22.0 \
    pyyaml>=6.0 \
    numpy>=1.24.0 \
    requests>=2.31.0 \
    pydantic>=2.0

COPY . .

EXPOSE 9000

CMD ["python", "server.py"]
