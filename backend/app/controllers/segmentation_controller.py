"""Handles segmentation API requests (e.g., POST image, GET result)."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.segmentation_service import SegmentationService

router = APIRouter()
segmentation_service = SegmentationService()


@router.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file locally or directly upload to a temporary S3 location
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Perform segmentation and get results
        result = segmentation_service.segment_image(file_location)
        return {"message": "Segmentation successful", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_segmentation_results(limit: int = 10):
    try:
        results = segmentation_service.repo.get_segmentation_results(limit)
        return {"segmentation_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
