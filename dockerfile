# Imagen base con Python
FROM python:3.11-slim

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Dependencias del sistema necesarias para osmnx / geopandas
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar requirements primero (mejor cache)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Puerto por defecto de Dash
EXPOSE 8050

# Variables necesarias para Dash
ENV HOST=0.0.0.0
ENV PORT=8050

# Comando de ejecución
CMD ["python", "web.py"]
