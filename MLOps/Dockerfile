FROM python:3.9-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file requirements
COPY requirements.txt .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice
COPY . .

# Esponi la porta
EXPOSE 5000

# Configura variabili ambiente per produzione
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Comando di avvio
CMD ["python", "app.py"] 