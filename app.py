from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import json
import gdown

# === Google Drive Model Setup ===
MODEL_FILE_ID = "10lbEfZFMQoOtVi2emipHsPzUACdPaifm"
MODEL_LOCAL_PATH = os.path.join("model", "plant_disease_cnn_custom.h5")
GDRIVE_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

# Ensure the folder exists
os.makedirs("model", exist_ok=True)

# Download model if not exists
if not os.path.exists(MODEL_LOCAL_PATH):
    print("[INFO] Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_LOCAL_PATH, quiet=False)
    print("[INFO] Model downloaded ✅")
else:
    print("[INFO] Model already exists locally ✅")

# === Load the Model ===
print("[INFO] Loading general model...")
general_model = load_model(MODEL_LOCAL_PATH)
print("[INFO] General model loaded ✅")

# === Load Class Names ===
with open(os.path.join("data", "class_names.json"), "r") as f:
    general_class_names = json.load(f)
print("[INFO] Class names loaded ✅")

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

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

        # For now, description and treatment are empty
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
