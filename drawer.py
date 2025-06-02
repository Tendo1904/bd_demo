import tkinter as tk
from PIL import Image, ImageDraw
import io
import base64
import requests
import threading

WIDTH, HEIGHT = 280, 280  # большое разрешение, потом уменьшим до 28x28


class PaintApp:
    """
    A digit drawing application that allows users to draw digits on a canvas
    and send them for prediction.

    Methods:
        __init__
        paint
        clear_canvas
        trigger_send_image
        send_image

    Attributes:
        root
        canvas
        predict_button
        clear_button
        result_label
        latest_request_id
        thread_lock

    Summary:
        The __init__ method initializes the application, setting up the user
        interface and necessary components for drawing and predicting digits.
        The paint method handles drawing on the canvas using mouse events.
        The clear_canvas method resets the drawing area and result display.
        The trigger_send_image method manages the threading aspects of sending
        an image for prediction, while the send_image method processes the image
        and communicates with a server to obtain digit predictions.
    """

    def __init__(self, root):
        """
        Initialize the digit drawing application.

            This method sets up the main window, including the canvas for drawing digits,
            buttons for predicting the digit and clearing the canvas, and a label for
            displaying prediction results. It also initializes necessary attributes for
            handling image drawing and threading.

            Args:
                root: The root window where the application will be displayed.

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
        Draws a black oval on the canvas at the position of the mouse event.

            This method is triggered by a mouse event and creates an oval shape on the
            canvas based on the mouse coordinates. The oval is filled with black color
            and has no outline.

            Args:
                event: An event object that contains the x and y coordinates of the mouse
                       pointer when the event occurred.

            Returns:
                None.
        """
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        """
        Clears the drawing canvas and resets UI elements.

            This method removes all drawings from the canvas, fills the canvas
            with a white rectangle to clear any visual artifacts, and resets
            the result label text to an empty string.

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
        Triggers sending an image in a separate thread.

            This method increments the latest request ID in a thread-safe manner,
            creates a new thread to send an image using the updated request ID,
            and starts the thread as a daemon.

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

            This method resizes an image to 28x28 pixels, inverts the colors,
            converts it to a PNG format, and sends it as a base64 encoded string
            to a specified server API for classification. It attempts to post the
            request to the last successful server first and then falls back to
            a second server if the first fails. The result of the prediction
            is then displayed on the application interface.

            Parameters:
                None

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
