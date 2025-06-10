import torch
from torchvision import transforms
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from model import SimpleCNN
import io
import base64


class MNISTHandler(BaseHandler):
    """
    A class to handle the MNIST dataset for model inference and preprocessing.

    This class is designed to manage the workflow of loading a pre-trained model,
    preprocessing input data, performing inference, and post-processing the output.

    Methods:
        __init__(): Initializes the model and device for processing.
        initialize(ctx): Loads pre-trained weights into the model.
        preprocess(data): Converts base64-encoded images into grayscale tensors.
        inference(input_tensor): Executes the model on an input tensor to predict class indices.
        postprocess(inference_output): Returns the inference output without modifications.

    Attributes:
        model: The convolutional neural network model used for inference, initialized to None.
        device: The computation device (GPU or CPU) determined for processing.
        transform: Transformations defined for preprocessing input images.

    This class encapsulates all the necessary functionalities to efficiently
    prepare input data, interact with a model, and obtain predictions while
    managing GPU or CPU resources effectively.
    """

    def __init__(self):
        """
        Initialize the model and device for processing.

            This method sets up the initial state of the class, including
            initializing the model to None, determining the computation
            device (GPU or CPU), and defining a series of image transformations
            that will be applied to input data.

            Parameters:
                None

            Returns:
                None
        """
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),  # just in case
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def initialize(self, ctx):
        """
        Initializes the model by loading pre-trained weights.

            This method loads a pre-trained model from the specified directory and
            prepares it for evaluation by setting the appropriate device and mode.

            Args:
                ctx: The context object that contains system properties, including
                     the path to the model directory.

            Returns:
                None

            The method uses the following imported methods:
            - `SimpleCNN`: This class defines a convolutional neural network.
              It is initialized in this method to create an instance of the model.
            - `torch.load`: This function is used to load the model weights from a
              file. In this case, it retrieves weights from the 'mnist_model.pth'
              file located in the `model_dir`.
        """
        # Load model
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        model_path = f"{model_dir}/mnist_model.pth"
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        """
        Preprocesses the input data to extract and transform an image.

            This method retrieves a base64-encoded image from the provided data and
            converts it into a grayscale tensor suitable for further processing.

            Args:
                data: A list of dictionaries containing image data. The first dictionary
                      must include either a "body" or "data" key, which contains a
                      base64-encoded representation of the image, or a key "image_bytes"
                      in case of a dictionary structure.

            Returns:
                A tensor representing the preprocessed grayscale image, ready for
                input into a machine learning model.

            Raises:
                ValueError: If the expected base64-encoded image data is not found
                            in the input `data`.
        """
        image_b64 = data[0].get("body") or data[0].get("data")
        if isinstance(image_b64, dict) and "image_bytes" in image_b64:
            image_bytes = base64.b64decode(image_b64["image_bytes"])
        else:
            raise ValueError("Expected base64-encoded image_bytes in JSON")

        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image

    def inference(self, input_tensor):
        """
        Perform inference on the given input tensor using the model.

            This method runs the model to produce output predictions for the
            given input tensor without tracking gradients. It retrieves and
            prints the predicted class index with the highest confidence.

            Args:
                input_tensor: The input tensor to be fed into the model for
                              inference.

            Returns:
                A list containing the predicted class index of the input tensor.
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            print(predicted.item())
        return [predicted.item()]

    def postprocess(self, inference_output):
        """
        Post-processes the inference output.

            This method receives the output from an inference process and returns it without any modifications.

            Args:
                inference_output: The output generated from the inference process.

            Returns:
                The unmodified inference output.
        """
        return inference_output
