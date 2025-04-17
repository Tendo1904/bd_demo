import tkinter as tk
from PIL import Image, ImageDraw
import io
import base64
import requests

WIDTH, HEIGHT = 280, 280  # большое разрешение, потом уменьшим до 28x28

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg='white')
        self.canvas.pack()

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.send_image)
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.result_label.pack()

        self.image = Image.new("L", (WIDTH, HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)

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

    def send_image(self):
        # Масштабируем до 28x28, как MNIST
        img = self.image.resize((28, 28)).convert("L")

        # Инвертируем (белый -> 0, черный -> 255)
        img = Image.eval(img, lambda x: 255 - x)

        # Отправляем в API
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        try:
            response = requests.post("http://localhost:8000/predict", json={"image_bytes": img_b64})
            pred = response.json()["class_id"]
            self.result_label.config(text=f"Predicted: {pred}")
        except Exception as e:
            self.result_label.config(text=f"Error: {e}")

# Запуск
if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()