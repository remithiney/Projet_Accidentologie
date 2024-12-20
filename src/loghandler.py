import logging
import queue
from logging.handlers import QueueHandler, QueueListener
from typing import List

class LogHandlerManager:
    def __init__(self, log_file_path: str):
        # Créer une queue pour les messages de log
        self.log_queue = queue.Queue()
        
        # Configurer le QueueHandler
        self.queue_handler = QueueHandler(self.log_queue)
        
        # Configurer un gestionnaire de fichier pour écrire les logs dans un fichier
        self.file_handler = logging.FileHandler(log_file_path)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(self.formatter)
        
        # Configurer le QueueListener pour écouter la queue et écrire les messages de log
        self.queue_listener = QueueListener(self.log_queue, self.file_handler)

    def start_listener(self):
        # Démarrer le QueueListener
        self.queue_listener.start()

    def stop_listener(self):
        # Arrêter le QueueListener
        self.queue_listener.stop()

    def get_queue_handler(self):
        # Retourner le QueueHandler pour être utilisé par d'autres loggers
        return self.queue_handler

    def configure_logger(self, loggers: List[logging.Logger]):
        # Ajouter le QueueHandler à la liste des loggers fournie
        for logger in loggers:
            logger.addHandler(self.queue_handler)
            logger.addHandler(self.file_handler)  # Ajouter également le FileHandler directement au logger
            logger.setLevel(logging.INFO)  # Passer le niveau à DEBUG pour capturer plus de détails
