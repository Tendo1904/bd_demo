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
    Represents a request to handle image data.

        This class is designed to encapsulate the image data to be processed.
        It contains the raw image bytes that can be utilized in various image handling operations.

        Attributes:
            image_bytes: The raw bytes of the image.

        Methods:
            (If applicable, a list of methods would be mentioned here.)

        The `image_bytes` attribute holds the binary representation of an image, which is essential
        for image processing tasks.
    """

    image_bytes: bytes


app = FastAPI()


@app.post("/predict")
async def predict(req: ImageRequest):
    """
    Predicts the class ID of an image from the request.

        This method handles a POST request that contains an image in base64 format,
        decodes it, processes it using a pre-defined transformation, and then uses
        a machine learning model to predict the class of the image.

        Args:
            req: An object that contains the base64-encoded image as a string.

        Returns:
            A dictionary containing the predicted class ID of the image.
    """
    image_bytes = base64.b64decode(req.image_bytes)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return {"class_id": predicted.item()}
