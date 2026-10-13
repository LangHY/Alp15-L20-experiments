# syntax=docker/dockerfile:1
ARG UV_VERSION=0.11.11
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

# 默认 CPU 工具镜像。只在用户手动 build 时拉取镜像及依赖。
FROM python:3.12.14-slim-bookworm AS cpu
COPY --from=uv-bin /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/workspace/scripts:/workspace/compression \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --extra analysis
COPY . .
RUN python tools/restore_packaged_manifests.py
CMD ["python", "--version"]

# L20/CUDA 路线；不是 Jetson 镜像，也不在 build 过程中下载模型或启动 GPU。
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS l20
COPY --from=uv-bin /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/workspace/scripts:/workspace/compression:/workspace/external/alpamayo1.5/src \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_EXTENSIONS_DIR=/tmp/torch-extensions \
    HF_HOME=/cache/huggingface
WORKDIR /workspace
RUN uv python install 3.12.14
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-default-groups --extra l20 --extra awq
# 默认仅准备依赖，FlashAttention 构建是用户显式选择的步骤。
ARG INSTALL_FLASH_ATTN=0
ARG MAX_JOBS=4
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
      MAX_JOBS="$MAX_JOBS" uv pip install --python /opt/venv/bin/python \
      flash-attn==2.8.3 --no-build-isolation; \
    fi
COPY . .
RUN /opt/venv/bin/python tools/restore_packaged_manifests.py
CMD ["python", "--version"]

# 未指定 --target 时只构建 CPU 路线。
FROM cpu AS default
