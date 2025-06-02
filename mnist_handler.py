import torch
from torchvision import transforms
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from model import SimpleCNN
import io
import base64


class MNISTHandler(BaseHandler):
    """
    Handles MNIST dataset interaction and model inference.

        This class is responsible for managing the process of loading a pre-trained
        model, preprocessing input images, running inference on those images, and
        post-processing the inference output. It provides a streamlined interface
        for performing these operations in a structured manner.

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

        The `model` attribute stores the instance of the neural network used for
        inference. The `device` attribute specifies whether to use CPU or CUDA for
        model computations. The `transform` attribute contains transformation steps
        for preprocessing input images before feeding them into the model.
    """

    def __init__(self):
        """
        Initializes the class.

            This method sets up the necessary attributes for the class, including
            the model, device configuration for CUDA, and a series of image
            transformation steps that will be applied to input data.

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

            This method sets up the model for inference by loading its state dictionary from a specified path.
            It expects the context to provide the necessary properties to locate the model weights.

            Args:
                ctx: The context object containing system properties. It is expected to provide a key `model_dir`
                     which indicates the directory where the model weights are stored.

            Returns:
                None: This method does not return any value. It prepares the model for evaluation by loading
                the pre-trained state and setting the model to evaluation mode.

            Notes:
                The method uses the `SimpleCNN` class to instantiate the model. The `initialize` method uses
                the `load_state_dict` function from PyTorch to load the model weights from a file named
                `mnist_model.pth`, located in the directory specified by `model_dir` in the context properties.
                It also uses `map_location` to ensure the model is loaded onto the correct device specified by
                the instance variable `self.device`.

                The model is set to evaluation mode by calling `self.model.eval()`, preparing it for inference.
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
        Preprocesses the input data to obtain a grayscale tensor representation of an image.

            This method extracts a base64-encoded image from the provided data, decodes it,
            converts it to a grayscale image, applies a predefined transformation, and prepares
            it as a tensor for further processing.

            Args:
                data: A list of dictionaries containing the image data, which must include
                      either a "body" or "data" key. The first dictionary in the list should
                      contain base64-encoded image data under the "image_bytes" key.

            Returns:
                A tensor representing the preprocessed grayscale image, ready for input into
                a model or further analysis.
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
        Runs inference on the given input tensor using the model.

            This method takes an input tensor, performs a forward pass through the model
            without tracking gradients, and retrieves the predicted class for the input.
            The predicted class is then printed and returned as a list containing that class.

            Args:
                input_tensor: The input tensor for which inference is to be performed.

            Returns:
                A list containing the predicted class as an integer.
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            print(predicted.item())
        return [predicted.item()]

    def postprocess(self, inference_output):
        """
        Post-processes the inference output.

            This method takes the raw output from the inference process
            and returns it as is, allowing for potential modifications
            or analysis in the future.

            Args:
                inference_output: The raw output generated from the inference.

            Returns:
                The processed inference output.
        """
        return inference_output
