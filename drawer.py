import tkinter as tk
from PIL import Image, ImageDraw
import io
import base64
import requests
import threading

WIDTH, HEIGHT = 280, 280  # большое разрешение, потом уменьшим до 28x28

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg='white')
        self.canvas.pack()

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.trigger_send_image)
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.result_label.pack()

        self.image = Image.new("L", (WIDTH, HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)

        self._lock = threading.Lock()
        self._latest_request_id = 0
        self.last_working_index = 0

        self.servers = [
            "http://localhost:8080/predictions/mnist",   # TorchServe
            "http://localhost:8000/predict"              # FastAPI
        ]

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, WIDTH, HEIGHT], fill="white")
        self.result_label.config(text="")

    def trigger_send_image(self):
        with self._lock:
            self._latest_request_id += 1
            request_id = self._latest_request_id

        thread = threading.Thread(target=self.send_image, args=(request_id,))
        thread.daemon = True
        thread.start()

    def send_image(self, request_id):
        # Масштабируем до 28x28, как MNIST
        img = self.image.resize((28, 28)).convert("L")

        # Инвертируем (белый -> 0, черный -> 255)
        img = Image.eval(img, lambda x: 255 - x)

        # Отправляем в API
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
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