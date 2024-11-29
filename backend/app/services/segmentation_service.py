import torch
import os
from backend.app.repositories.segmentation_repo import SegmentationRepository
from backend.app.utils.s3_client import S3Client
from backend.models.vit_segmentation import VisionTransformerSegmentation
from backend.scripts.image_proccessing import preprocess_image  # Import preprocessing function


def _save_output_image(output):
    """
    Save the segmentation output to an image file.
    You might want to save the output tensor as a .png or .jpg file.
    """
    output_image = output.squeeze().cpu().numpy()  # Remove batch dimension
    output_image_path = "output/segmented_image.png"
    # Save the image as a PNG (you can use other formats like .jpg, etc.)
    from matplotlib import pyplot as plt
    plt.imsave(output_image_path, output_image, cmap="jet")
    return output_image_path


class SegmentationService:
    def __init__(self, model_path="models/vit_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionTransformerSegmentation().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.repo = SegmentationRepository()
        self.s3_client = S3Client()

    def segment_image(self, image_path):
        # Load and preprocess the image using the script's function
        preprocessed_image = preprocess_image(image_path)

        # Perform segmentation with the model
        with torch.no_grad():
            output = self.model(preprocessed_image.to(self.device))

        # Save the segmented image and upload to S3
        segmented_image_path = _save_output_image(output)
        segmented_image_url = self.s3_client.upload_file(segmented_image_path, os.path.basename(segmented_image_path))

        # Log the result in the database
        result_id = self.repo.save_segmentation_result(image_path, segmented_image_url)
        return {"result_id": result_id, "segmented_image_url": segmented_image_url}
