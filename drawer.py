import tkinter as tk
from PIL import Image, ImageDraw
import io
import base64
import requests
import threading

WIDTH, HEIGHT = 280, 280  # большое разрешение, потом уменьшим до 28x28


class PaintApp:
    """
    A simple application for drawing digits and predicting them using a server.

    This class provides a graphical interface for users to draw digit shapes
    on a canvas. It includes functionalities to predict the drawn digit, clear
    the canvas, and handle the communication with a server for digit recognition.

    Methods:
        __init__
        paint
        clear_canvas
        trigger_send_image
        send_image

    Attributes:
        None

    Method and Attribute Summary:
        - __init__: Initializes the application interface with a canvas, buttons,
          and a label for displaying predictions, along with thread safety mechanisms.
        - paint: Handles mouse events to draw oval shapes on the canvas.
        - clear_canvas: Resets the canvas to its initial state and clears the result label.
        - trigger_send_image: Starts a new thread to send the drawn image for prediction.
        - send_image: Processes and sends the drawn image to a server for prediction
          and updates the UI with the result.
    """

    def __init__(self, root):
        """
        Initializes the application interface for drawing a digit.

            This method sets up the main application window, including a canvas
            for drawing, buttons for predicting the digit and clearing the canvas,
            and a label to display the prediction results. It also initializes an image
            object for storing the drawn digit and sets up a lock for thread safety.

            Args:
                root: The root window or parent widget for the application interface.

            Returns:
                None
        """
        self.root = root
        self.root.title("Draw a Digit")
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg="white")
        self.canvas.pack()

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.predict_btn = tk.Button(
            self.button_frame, text="Predict", command=self.trigger_send_image
        )
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        self.clear_btn = tk.Button(
            self.button_frame, text="Clear", command=self.clear_canvas
        )
        self.clear_btn.pack(side=tk.LEFT)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.result_label.pack()

        self.image = Image.new("L", (WIDTH, HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)

        self._lock = threading.Lock()
        self._latest_request_id = 0
        self.last_working_index = 0

        self.servers = [
            "http://localhost:8080/predictions/mnist",  # TorchServe
            "http://localhost:8000/predict",  # FastAPI
        ]

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        """
        Draws an oval shape on the canvas based on mouse click coordinates.

            This method is triggered by a mouse event and it creates an oval on
            the canvas at the location where the mouse is clicked. The oval has a
            fixed size and is filled with a black color.

            Args:
                event: The mouse event containing the x and y coordinates
                       of the mouse click.

            Returns:
                None
        """
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        """
        Clears the canvas and resets its state.

            This method deletes all items from the canvas and redraws a white rectangle
            over the entire canvas area to clear it. Additionally, it resets the text
            of the result label to an empty string.

            Parameters:
                None

            Returns:
                None
        """
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, WIDTH, HEIGHT], fill="white")
        self.result_label.config(text="")

    def trigger_send_image(self):
        """
        Triggers the sending of an image in a separate thread.

            This method increments the latest request ID and starts a new thread
            to send an image using the updated request ID. The thread is set as a
            daemon, meaning it will not prevent the program from exiting.

            Parameters:
                None

            Returns:
                None
        """
        with self._lock:
            self._latest_request_id += 1
            request_id = self._latest_request_id

        thread = threading.Thread(target=self.send_image, args=(request_id,))
        thread.daemon = True
        thread.start()

    def send_image(self, request_id):
        """
        Sends a pre-processed image to a server for prediction.

            This method rescales the image to 28x28 pixels and converts it to grayscale,
            inverts the pixel values, and encodes the image as a base64 string. It then
            sends this encoded image to the preferred server for prediction. In case of
            a failure to connect, it retries with a fallback server. The method updates
            the user interface with the predicted class or an error message accordingly.

            Args:
                request_id: An identifier for the request, used to track the most
                    recent processing request and update the label correctly.

            Returns:
                None
        """
        # Масштабируем до 28x28, как MNIST
        img = self.image.resize((28, 28)).convert("L")

        # Инвертируем (белый -> 0, черный -> 255)
        img = Image.eval(img, lambda x: 255 - x)

        # Отправляем в API
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        payload = {"image_bytes": img_b64}

        def update_label(text):
            self.root.after(0, self.result_label.config, {"text": text})

        # Prioritize the last successful server
        primary = self.servers[self.last_working_index]
        fallback = self.servers[1 - self.last_working_index]

        for idx, server in enumerate([primary, fallback]):
            try:
                response = requests.post(server, json=payload, timeout=3)
                pred = response.json()
                if isinstance(pred, dict):
                    pred = pred.get("class_id", pred)

                with self._lock:
                    if request_id == self._latest_request_id:
                        self.last_working_index = self.servers.index(server)
                        update_label(f"Predicted: {pred}")
                return
            except Exception:
                continue

        with self._lock:
            if request_id == self._latest_request_id:
                update_label("Error: all servers unreachable.")


# Запуск
if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
