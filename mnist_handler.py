import torch
from torchvision import transforms
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from model import SimpleCNN
import io
import base64

class MNISTHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # just in case
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def initialize(self, ctx):
        # Load model
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        model_path = f"{model_dir}/mnist_model.pth"
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        image_b64 = data[0].get("body") or data[0].get("data")
        if isinstance(image_b64, dict) and "image_bytes" in image_b64:
            image_bytes = base64.b64decode(image_b64["image_bytes"])
        else:
            raise ValueError("Expected base64-encoded image_bytes in JSON")
        
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image

    def inference(self, input_tensor):
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            print(predicted.item())
        return [predicted.item()]

    def postprocess(self, inference_output):
        return inference_output
