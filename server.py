from fastapi import FastAPI
from pydantic import BaseModel
from model import SimpleCNN
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

model = SimpleCNN()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class ImageRequest(BaseModel):
    image_bytes: bytes

app = FastAPI()

@app.post("/predict")
async def predict(req: ImageRequest):
    image_bytes = base64.b64decode(req.image_bytes)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return {"class_id": predicted.item()}