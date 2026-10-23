# ---- build stage: install dependencies into a virtualenv
FROM python:3.12-slim AS build
WORKDIR /app
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- runtime stage: small image, non-root user
FROM python:3.12-slim
WORKDIR /app
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY --from=build /venv /venv
COPY app ./app
COPY data ./data
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
