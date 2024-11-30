FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir qiskit qiskit-aer

WORKDIR /usr/src/app

COPY main.py .
COPY file.qasm .

CMD ["/bin/bash"]

