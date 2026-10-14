FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG INSTALL_RL=false
COPY pyproject.toml LICENSE README.md ./
COPY src ./src
RUN if [ "$INSTALL_RL" = "true" ]; then python -m pip install --no-cache-dir '.[rl]'; \
    else python -m pip install --no-cache-dir .; fi
COPY deploy.py eval.py train.py benchmark.py serve.py ./
COPY instances ./instances
EXPOSE 8080
# Binds externally only inside container; reverse proxy should enforce auth and TLS.
CMD ["python", "serve.py", "--host", "0.0.0.0", "--strategy", "lookahead"]
