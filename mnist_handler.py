import torch
from torchvision import transforms
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from model import SimpleCNN
import io
import base64


class MNISTHandler(BaseHandler):
    """
    A class for handling MNIST image processing and model inference.

    This class is responsible for preprocessing MNIST images, performing
    inference with a trained model, and post-processing the model's
    outputs. It initializes the model on the appropriate computational
    device and provides methods for loading the model, processing images,
    and obtaining predictions.

    Methods:
        __init__
        initialize
        preprocess
        inference
        postprocess

    Attributes:
        model
        device
        transform

    - model: Represents the neural network model to be used for inference.
    - device: The computational device (CPU or GPU) for running the model.
    - transform: A series of transformations applied to the input images
      before feeding them to the model.

    Each method in the class facilitates specific tasks such as loading
    the model's weights from a specified directory, preprocessing image
    data, running predictions, and handling the outputs of the inference
    process.
    """

    def __init__(self):
        """
        Initialize the model and device for the class.

            This method sets up the initial state of the object by calling the
            constructor of the parent class. It initializes the model to None
            and determines the appropriate device (CPU or GPU) for computation
            based on the availability of CUDA. Additionally, it configures a
            series of image transformations which include conversion to grayscale,
            resizing to 28x28 pixels, conversion to a tensor, and normalization.

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
        Initializes the model by loading its state from a specified directory.

            This method retrieves the model directory from the provided context, constructs
            the full path to the model file, and initializes the SimpleCNN model. It
            then loads the model's state from the specified file and sets the model to evaluation mode.

            Args:
                ctx: The context object that contains system properties, specifically
                     the directory where the model is stored.

            Returns:
                None: This method does not return a value.

            Usage:
                - The method assumes the existence of a directory specified by
                  `model_dir` within the system properties of `ctx`.
                - The model is loaded using PyTorch’s `torch.load` method, which allows
                  the loading of the model state dictionary from the specified file path.
                - The model is an instance of the SimpleCNN class, which is defined
                  by its initializer (`__init__`). The model comprises a sequence of
                  convolutional and fully connected layers, programmed to process
                  MNIST images.
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
        Preprocesses the input data to extract and convert an image.

            This method extracts a base64-encoded image from the provided data,
            decodes it, converts it to grayscale, and applies a transformation
            before returning the processed image.

            Args:
                data: A list containing a dictionary with either a "body"
                      or "data" key, which must contain a base64-encoded
                      image embedded in an "image_bytes" field.

            Returns:
                A tensor representing the processed grayscale image after
                transformation and moved to the device specified in the
                class attributes.

            Raises:
                ValueError: If the expected base64-encoded image_bytes is
                            not found in the input data.
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
        Runs inference on the given input tensor and prints the predicted class.

            This method uses a pre-trained model to make predictions on the provided
            input tensor without calculating gradients. It retrieves the predicted
            class with the highest score from the model's output.

            Args:
                input_tensor: The input data to the model for which predictions are to be made.

            Returns:
                A list containing the predicted class label as an integer.
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            print(predicted.item())
        return [predicted.item()]

    def postprocess(self, inference_output):
        """
        Post-processes the inference output.

            This method takes the output from a model inference and returns it directly.
            It can be extended to include additional processing steps as needed.

            Args:
                inference_output: The output from the model inference.

            Returns:
                The processed inference output.
        """
        return inference_output
