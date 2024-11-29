"""Handles training-related requests (e.g., POST training data)."""

# backend/app/controllers/training_controller.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from backend.services.training_service import TrainingService
from backend.db.db import get_db
from sqlalchemy.orm import Session
import os

router = APIRouter()


# Pydantic model to receive training parameters
class TrainRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4


@router.post("/train")
async def start_training(
        train_request: TrainRequest,
        db: Session = Depends(get_db)
):
    try:
        # Initialize training service with the parameters
        training_service = TrainingService(db, train_request.epochs, train_request.batch_size,
                                           train_request.learning_rate)

        # Start the training process (this can be a background task as well)
        training_service.start_training()

        return {"message": "Training started successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
