from fastapi import FastAPI
from backend.app.controllers.segmentation_controller import router as segmentation_router
from backend.app.controllers.training_controller import router as training_router
from backend.app.controllers.user_controller import router as user_router
from backend.app.db.db import init_db
import uvicorn


def create_app():
    # Initialize the FastAPI app
    app = FastAPI()

    # Include the routers (controllers) for different functionality
    app.include_router(segmentation_router, prefix="/segmentation", tags=["segmentation"])
    app.include_router(training_router, prefix="/training", tags=["training"])
    app.include_router(user_router, prefix="/user", tags=["user"])

    # Initialize the database
    init_db()

    return app


# Create the FastAPI app
app = create_app()

if __name__ == "__main__":
    # Run the application with Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
