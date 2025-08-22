from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import json
import requests
from tqdm import tqdm

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

# === Hugging Face Model Link ===
MODEL_FILE_NAME = "plant_disease_cnn_custom.h5"
MODEL_LOCAL_PATH = os.path.join(MODEL_FILE_NAME)
HUGGINGFACE_URL = "https://huggingface.co/IvarOp/pavaman-plantcare-backend/resolve/main/plant_disease_cnn_custom.h5"

# === Ensure model exists locally ===
def download_model(url, destination):
    if os.path.exists(destination):
        print("[INFO] Model already exists locally ✅")
        return

    print("[INFO] Downloading model from Hugging Face...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc="Downloading model",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print("[INFO] Model downloaded ✅")

download_model(HUGGINGFACE_URL, MODEL_LOCAL_PATH)

# === Load General Model ===
print("[INFO] Loading general model...")
general_model = load_model(MODEL_LOCAL_PATH)
print("[INFO] General model loaded ✅")

# === Dummy Class Names ===
# Replace this with your actual class_names.json contents if available
general_class_names = {
  "0": "Apple - Apple Scab",
  "1": "Apple - Black Rot",
  "2": "Apple - Cedar Apple Rust",
  "3": "Apple - Healthy",
  "4": "Blueberry - Healthy",
  "5": "Cherry - Powdery Mildew",
  "6": "Cherry - Healthy",
  "7": "Corn - Cercospora Leaf Spot (Gray Leaf Spot)",
  "8": "Corn - Common Rust",
  "9": "Corn - Northern Leaf Blight",
  "10": "Corn - Healthy",
  "11": "Grape - Black Rot",
  "12": "Grape - Esca (Black Measles)",
  "13": "Grape - Leaf Blight (Isariopsis Leaf Spot)",
  "14": "Grape - Healthy",
  "15": "Orange - Haunglongbing (Citrus Greening)",
  "16": "Peach - Bacterial Spot",
  "17": "Peach - Healthy",
  "18": "Pepper, Bell - Bacterial Spot",
  "19": "Pepper, Bell - Healthy",
  "20": "Potato - Early Blight",
  "21": "Potato - Late Blight",
  "22": "Potato - Healthy",
  "23": "Raspberry - Healthy",
  "24": "Soybean - Healthy",
  "25": "Squash - Powdery Mildew",
  "26": "Strawberry - Leaf Scorch",
  "27": "Strawberry - Healthy",
  "28": "Tomato - Bacterial Spot",
  "29": "Tomato - Early Blight",
  "30": "Tomato - Late Blight",
  "31": "Tomato - Leaf Mold",
  "32": "Tomato - Septoria Leaf Spot",
  "33": "Tomato - Spider Mites (Two-Spotted Spider Mite)",
  "34": "Tomato - Target Spot",
  "35": "Tomato - Yellow Leaf Curl Virus",
  "36": "Tomato - Mosaic Virus",
  "37": "Tomato - Healthy"
}


# === Routes ===
@app.route("/", methods=["GET"])
def health():
    return "✅ Pavaman PlantCare backend is running."

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['file']
        img_pil = Image.open(file).convert("RGB")

        img = img_pil.resize((224, 224))
        input_tensor = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = general_model.predict(input_tensor)[0]
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = general_class_names[str(idx)]
        print(f"[GENERAL] {label} ({confidence:.2f})")

        return jsonify({
            "disease": label,
            "confidence": round(confidence * 100, 2),
            "description": "",
            "treatment": ""
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
