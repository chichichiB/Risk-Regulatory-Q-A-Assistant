FROM python:3.12-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
RUN pip install --no-cache-dir uv==0.12.23
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra models --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
COPY corpus ./corpus
COPY models ./models
COPY eval ./eval
RUN uv sync --frozen --extra models --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "risk_qa.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
