FROM python:3.13-slim

WORKDIR /usr/src/app

COPY api_services/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Ricorda di copiare sia il codice dell'API sia la cartella `shared` e di usare `gunicorn` come `CMD`.
COPY api_services/ .
COPY shared ./shared

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "api:app"]


