FROM python:3.12.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RESEARCH_GAP_AUTO_MIGRATE=false

WORKDIR /app
RUN addgroup --system researchgap && adduser --system --ingroup researchgap researchgap
COPY requirements.lock ./
RUN python -m pip install --upgrade pip==25.2 && python -m pip install -r requirements.lock
COPY --chown=researchgap:researchgap src ./src
COPY --chown=researchgap:researchgap main.py IMPORTANT.md README.md ./
RUN mkdir -p /app/data/cache && chown -R researchgap:researchgap /app/data

USER researchgap
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)" || exit 1
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
