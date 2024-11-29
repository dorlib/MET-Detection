# backend/app/repositories/training_repo.py

from sqlalchemy.orm import Session
from backend.db.models import TrainingLog
from datetime import datetime


class TrainingRepo:
    def __init__(self, db: Session):
        self.db = db

    def save_training_log(self, status: str, epoch: int, loss: float):
        log = TrainingLog(status=status, epoch=epoch, loss=loss, timestamp=datetime.now())
        self.db.add(log)
        self.db.commit()

    def get_training_logs(self):
        return self.db.query(TrainingLog).all()
