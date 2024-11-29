# backend/db/models.py

from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from base import Base


class TrainingLog(Base):
    __tablename__ = 'training_logs'

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, index=True)
    epoch = Column(Integer)
    loss = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TrainingLog(status={self.status}, epoch={self.epoch}, loss={self.loss}, timestamp={self.timestamp})>"
