"""API routes for model inference, training, etc."""

from fastapi import APIRouter, UploadFile, File
from backend.app.services.segmentation_service import SegmentationService
from backend.app.services.training_service import TrainingService

router = APIRouter()

segmentation_service = SegmentationService()
training_service = TrainingService()


@router.post("/segment")
async def segment_image(file: UploadFile, user_id: int):
    image = Image.open(file.file)
    result = segmentation_service.predict_and_save(image, user_id)
    return {"segmentation_result": result}


@router.post("/train")
async def train_model():
    training_service.train(dataset="your_dataset_path")
    return {"status": "Training started"}
