from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image   # <-- using Pillow

# --- Setup Flask ---
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Load Model & Scaler ---
MODEL_PATH = "autism_multimodal_model.keras"
SCALER_PATH = "meta_scaler.pkl"

model = load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

IMG_HEIGHT, IMG_WIDTH = 128, 128

# --- Helper function ---
def preprocess(img_path, age, gender, cars):
    # Load image in grayscale with PIL
    img = Image.open(img_path).convert("L")  
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  
    img = np.array(img) / 255.0  
    img = np.expand_dims(img, axis=(0, -1))  # shape (1,128,128,1)

    meta = np.array([[age, gender, cars]], dtype=np.float32)
    meta = scaler.transform(meta)

    return img, meta

# --- Routes (same as before) ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        age = float(request.form["age"])
        gender = int(request.form["gender"])  # 1=Male, 0=Female
        cars = float(request.form["cars"])

        if file.filename == "":
            return redirect(request.url)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Preprocess & Predict
        img, meta = preprocess(filepath, age, gender, cars)
        pred = model.predict([img, meta])
        prob = float(pred[0][0])
        result = "Autism (TS)" if prob > 0.5 else "Control (TC)"

        return render_template("index.html", 
                               prediction=result, 
                               probability=f"{prob:.4f}")

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
