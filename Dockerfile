# Control plane: REST API, web UI, MCP server and the ML Assistant in one process.
# The SageMaker stage image (ML_FACTORY_STAGE_IMAGE_URI) is a separate image and is not built here.
# Python 3.12, not 3.10 as in Dockerfile-example: pyproject.toml requires >=3.12.
FROM python:3.12-slim-bookworm

ARG http_proxy=http://10.0.139.93:8080
ARG https_proxy=http://10.0.139.93:8080
ARG HTTP_PROXY=http://10.0.139.93:8080
ARG HTTPS_PROXY=http://10.0.139.93:8080

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
# Keep calls to the container itself off the proxy.
ENV NO_PROXY=localhost,127.0.0.1
ENV no_proxy=localhost,127.0.0.1

WORKDIR /app

RUN echo 'Acquire::http::Proxy "http://10.0.139.93:8080";' > /etc/apt/apt.conf.d/99proxy && \
    echo 'Acquire::https::Proxy "http://10.0.139.93:8080";' >> /etc/apt/apt.conf.d/99proxy

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY ml_engine ./ml_engine
COPY jobs ./jobs
COPY backend ./backend
COPY mcp_server ./mcp_server
COPY ml_factory_cli ./ml_factory_cli
COPY scripts ./scripts

RUN pip install --no-cache-dir ".[backend,mcp,cli,models,dictionary]"

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "mcp_server.app"]
