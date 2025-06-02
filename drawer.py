import tkinter as tk
from PIL import Image, ImageDraw
import io
import base64
import requests
import threading

WIDTH, HEIGHT = 280, 280  # большое разрешение, потом уменьшим до 28x28


class PaintApp:
    """
    A simple drawing application that allows users to draw on a canvas
    and send their drawings for prediction.

    Methods:
        __init__
        paint
        clear_canvas
        trigger_send_image
        send_image

    Attributes:
        root
        canvas
        result_label
        request_id

    The class provides functionality for drawing on the canvas via mouse events,
    clearing the canvas, and sending images to a prediction server. It initializes
    the main application window, handles user inputs for drawing, manages
    the sending of images asynchronously, and displays prediction results.
    Each image sent for prediction is processed to meet server requirements,
    ensuring a user-friendly experience.
    """

    def __init__(self, root):
        """
        Initializes the Drawing Application.

                This method sets up the main application window, including the title,
                canvas for drawing, control buttons for prediction and clearing the canvas,
                and a label for displaying results. It also initializes necessary attributes
                for image handling and communication with prediction servers.

                Args:
                    root: The main window or root widget for the application.

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
        Draws a black oval on the canvas at the position of a mouse event.

            This method is triggered by a mouse event and creates an oval on the
            canvas at the coordinates where the event occurred. The size of the
            oval is fixed and determined by an offset of 8 pixels in each direction
            from the event's x and y coordinates.

            Args:
                event: An object containing the mouse event data, including
                the x and y coordinates.

            Returns:
                None
        """
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        """
        Clears the drawing canvas and resets the result label.

            This method deletes all existing drawings on the canvas and fills
            the entire canvas area with white color. It also clears the text
            displayed in the result label.

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
        Initiates the process to send an image in a separate thread.

            This method increments the request ID to ensure that each image
            sending request is unique. It then starts a new thread that calls
            the `send_image` method with the generated request ID.

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
        Sends a processed image to a server for prediction.

            This method resizes an image to 28x28 pixels, inverts its colors,
            and sends it to a predefined server API. It prioritizes the last
            successfully reached server and handles communication errors gracefully.

            Args:
                request_id: An identifier for the request, used to manage
                             the state of the request and ensure that
                             the most recent response is considered.

            Returns:
                None: This method does not return a value. It updates the
                      user interface with the prediction result or an error
                      message, depending on the response from the server.
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
