from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import json
import google.generativeai as genai

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

# === Load General Model ===
GENERAL_MODEL_PATH = os.path.join("model", "plant_disease_cnn_custom.h5")
print("[INFO] Loading general model...")
general_model = load_model(GENERAL_MODEL_PATH)
print("[INFO] General model loaded ✅")

# === Load Class Names ===
with open(os.path.join("data", "class_names.json"), "r") as f:
    general_class_names = json.load(f)
print("[INFO] Class names loaded ✅")

# === Gemini API Setup ===
genai.configure(api_key="YOUR_GEMINI_API_KEY")
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

        # Preprocess for general model
        img = img_pil.resize((224, 224))
        input_tensor = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = general_model.predict(input_tensor)[0]
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = general_class_names[str(idx)]
        print(f"[GENERAL] {label} ({confidence:.2f})")

        description = query_gemini(get_description_prompt(label))

        return jsonify({
            "disease": label,
            "confidence": round(confidence * 100, 2),
            "description": description,
            "treatment": ""  # Can use /get-treatment route if needed
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
    app.run(debug=True, host="0.0.0.0", port=5000)
