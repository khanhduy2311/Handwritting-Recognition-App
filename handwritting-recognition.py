import os
import io
import base64
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from PIL import Image, ImageOps
import tensorflow as tf

# Model setup
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Train or load model
MODEL_PATH = "mnist_cnn.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = create_model()
    model.fit(x_train, y_train, epochs=3, validation_split=0.1, verbose=1)
    model.save(MODEL_PATH)

# UI

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Digit Recognizer</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #74ebd5 0%, #9face6 100%);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      margin: 0;
      color: #333;
    }
    h2 {
      font-size: 28px;
      margin-bottom: 20px;
      color: #222;
      text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    #canvas {
      border: 2px solid #444;
      border-radius: 16px;
      background: #fff;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      cursor: crosshair;
    }
    .btn {
      margin: 15px 10px;
      padding: 12px 24px;
      font-size: 16px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-weight: bold;
      color: #fff;
      background: #4a90e2;
      transition: all 0.3s ease;
    }
    .btn:hover {
      background: #357ABD;
      transform: scale(1.05);
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    #result {
      margin-top: 20px;
      font-size: 24px;
      font-weight: bold;
      color: #222;
      padding: 12px 20px;
      border-radius: 12px;
      background: rgba(255,255,255,0.8);
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      display: inline-block;
      min-width: 200px;
    }
  </style>
</head>
<body>
  <h2>🖌️ Draw a digit (0–9)</h2>
  <canvas id="canvas" width="280" height="280"></canvas><br>
  <button class="btn" onclick="clearCanvas()">🧹 Clear</button>
  <button class="btn" onclick="predict()">🔮 Predict</button>
  <div id="result">Prediction: ...</div>

  <script>
    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let drawing = false;
    canvas.addEventListener('mousedown', e => { drawing = true; draw(e); });
    canvas.addEventListener('mouseup', e => { drawing = false; ctx.beginPath(); });
    canvas.addEventListener('mousemove', draw);

    function draw(e) {
      if (!drawing) return;
      ctx.lineWidth = 20;
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'black';
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(e.offsetX, e.offsetY);
    }

    function clearCanvas() {
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      document.getElementById('result').innerText = 'Prediction: ...';
    }

    function predict() {
      let dataURL = canvas.toDataURL('image/png');
      fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
      })
      .then(res => res.json())
      .then(data => {
        document.getElementById('result').innerText = 'Prediction: ' + data.prediction;
      });
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img_data = data['image'].split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    pred = model.predict(arr)[0]
    digit = int(np.argmax(pred))
    return jsonify({"prediction": digit})

if __name__ == "__main__":
    app.run(debug=True)
