from fastapi import FastAPI
from pydantic import BaseModel
from model import SimpleCNN
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN()
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


class ImageRequest(BaseModel):
    """
    Handles the processing and storage of image data.

        This class is responsible for managing the image data in byte format,
        allowing for operations related to image requests.

        Attributes:
            image_bytes: The byte representation of the image data.

        Methods:
            (no methods defined in this class)
    """

    image_bytes: bytes


app = FastAPI()


@app.post("/predict")
async def predict(req: ImageRequest):
    """
    Handles image prediction requests.

        This method decodes an incoming base64-encoded image, processes it,
        and passes it through a machine learning model to obtain a predicted
        class. The predicted class ID is then returned as part of the response.

        Args:
            req: An object containing the base64-encoded image data to be processed.

        Returns:
            A dictionary containing the predicted class ID as "class_id".
    """
    image_bytes = base64.b64decode(req.image_bytes)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return {"class_id": predicted.item()}
