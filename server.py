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
    A class to handle image requests, encapsulating the functionality
    related to image data processing and management.

    Attributes:
        image_bytes: The raw byte data of the image.

    Methods:
        None

    This class is designed to receive, store, and facilitate operations
    on image data, providing a simple interface for working with images
    in byte form.
    """

    image_bytes: bytes


app = FastAPI()


@app.post("/predict")
async def predict(req: ImageRequest):
    """
    Predicts the class of an image based on the input data.

    This method decodes a base64-encoded image, processes it, and then uses a
    pre-trained model to predict the class of the image. The prediction is
    returned as a dictionary containing the class identifier.

    Args:
        req: An instance of ImageRequest containing the base64-encoded image data.

    Returns:
        A dictionary with the predicted class identifier under the key "class_id".
    """
    image_bytes = base64.b64decode(req.image_bytes)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return {"class_id": predicted.item()}
