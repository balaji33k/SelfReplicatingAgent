FROM python:3.11-slim

# HuggingFace Spaces runs as UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy all project files
COPY --chown=user:user . .

# Create writable directories for evolution data
RUN mkdir -p data generations/gen_1/results generations/gen_1/sandbox \
    && chmod -R 777 data generations

# HuggingFace Spaces requires port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

USER user

CMD ["python", "serve_railway.py"]
