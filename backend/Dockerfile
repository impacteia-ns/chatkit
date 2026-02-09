FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema (mínimo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copia arquivos de dependências primeiro (para cache)
COPY pyproject.toml ./

# Instala o projeto (usa pyproject)
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .

# Copia o restante do código
COPY . .

# Railway expõe a porta via variável PORT
ENV PORT=8000
EXPOSE 8000

# Ajuste comum: tenta iniciar via uvicorn (muito provável nesse template)
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
