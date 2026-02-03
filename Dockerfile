FROM python:3.11-slim
WORKDIR /app

# Installation des dépendances système (gcc pour la compilation et libmagic pour le format des fichiers)
RUN apt-get update && apt-get install -y \
    gcc \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Préparation du dossier pour stocker les modèles d'IA
ENV HF_HOME=/app/models_cache
RUN mkdir -p /app/models_cache

# Définition du PYTHONPATH pour que Python trouve le dossier 'rag'
COPY . .
ENV PYTHONPATH=/app

# Exécution du main qui est dans le dossier app
CMD ["python", "app/main.py"]