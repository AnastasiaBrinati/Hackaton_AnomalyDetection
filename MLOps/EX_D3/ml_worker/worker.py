# Lo scopo dello script è importare l'istanza di `celery_app` dalla configurazione condivisa 
# in modo che il comando di avvio del worker possa trovarla.

from shared.celery_config import celery_app
