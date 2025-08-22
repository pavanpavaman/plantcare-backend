from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import json
import google.generativeai as genai
import cv2

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

# === Load Models ===
GENERAL_MODEL_PATH = os.path.join("model", "plant_disease_cnn_custom.h5")
MANGO_MODEL_PATH = os.path.join(r"C:\Users\Pavan\Pavaman Tech\disease_detection\pavaman-plantcare-farm\backend\model\mango_leaf_disease_model.keras")

print("[INFO] Loading general model...")
general_model = load_model(GENERAL_MODEL_PATH)
print("[INFO] General model loaded ✅")

print("[INFO] Loading mango model...")
mango_model = load_model(MANGO_MODEL_PATH)
print("[INFO] Mango model loaded ✅")

# === Load Class Names ===
with open(os.path.join("data", "class_names.json"), "r") as f:
    general_class_names = json.load(f)

with open(os.path.join("data", "mango_class_names.json"), "r") as f:
    mango_class_names = json.load(f)

print("[INFO] Class names loaded ✅")

# === Gemini API Setup ===
genai.configure(api_key="AIzaSyBupvmvDpajay4jPAkAqbgbjFqsR7ZsZJE")
llm = genai.GenerativeModel("gemini-1.5-flash")

def get_description_prompt(label):
    return f"Give a short scientific description of the plant disease '{label}' in 3–4 lines. Use clean Markdown formatting."

def get_prevention_prompt(label):
    return f"Suggest 3–5 bullet-point preventive measures for the plant disease '{label}'. Use clear Markdown formatting."

def get_treatment_prompt(label):
    return f"Suggest 3–5 bullet-point treatment methods for the plant disease '{label}'. Use clear Markdown formatting."

def query_gemini(prompt):
    try:
        response = llm.generate_content(prompt)
        return response.text.strip() if response.text.strip() else "⚠️ No valid response from Gemini."
    except Exception as e:
        print("Gemini LLM error:", e)
        return "⚠️ LLM failed to respond."

# === Mango Preprocessing ===
def preprocess_mango_image(img_pil, target_size=(224, 224)):
    img = np.array(img_pil)
    img = cv2.resize(img, (256, 256))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(img, img, mask=mask)

    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    final = cv2.resize(enhanced, target_size)
    return final / 255.0

# === Routes ===
@app.route("/", methods=["GET"])
def health():
    return "✅ Pavaman PlantCare backend is running."

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    model_type = request.form.get("model_type", "general").lower()
    if model_type not in ["general", "mango"]:
        return jsonify({"error": "Invalid model type. Choose 'general' or 'mango'."}), 400

    try:
        file = request.files['file']
        img_pil = Image.open(file).convert("RGB")

        if model_type == "general":
            img = img_pil.resize((224, 224))
            input_tensor = np.expand_dims(np.array(img) / 255.0, axis=0)
            preds = general_model.predict(input_tensor)[0]
            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            label = general_class_names[str(idx)]
            print(f"[GENERAL] {label} ({confidence:.2f})")

        elif model_type == "mango":
            input_tensor = np.expand_dims(preprocess_mango_image(img_pil), axis=0)
            preds = mango_model.predict(input_tensor)[0]
            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            label = mango_class_names[str(idx)]
            print(f"[MANGO] {label} ({confidence:.2f})")

        description = query_gemini(get_description_prompt(label))

        return jsonify({
            "disease": label,
            "confidence": round(confidence * 100, 2),
            "description": description,
            "treatment": ""
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/get-prevention", methods=["POST"])
def get_prevention():
    try:
        data = request.get_json()
        label = data.get("label")
        if not label:
            return jsonify({"error": "Missing disease label"}), 400
        return jsonify({"prevention": query_gemini(get_prevention_prompt(label))})
    except Exception as e:
        print("LLM prevention error:", e)
        return jsonify({"prevention": "⚠️ Error fetching prevention info."})

@app.route("/get-treatment", methods=["POST"])
def get_treatment():
    try:
        data = request.get_json()
        label = data.get("label")
        if not label:
            return jsonify({"error": "Missing disease label"}), 400
        return jsonify({"treatment": query_gemini(get_treatment_prompt(label))})
    except Exception as e:
        print("LLM treatment error:", e)
        return jsonify({"treatment": "⚠️ Error fetching treatment."})

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)
