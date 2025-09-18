# app.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = {'png','jpg','jpeg','tif','tiff'}
MODEL_PATH = 'model.h5'   # change to your model path

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once at start
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# Example labels -- replace with your classes / order used during training
labels = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Age-related Macular Degeneration", "Other"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def preprocess_image(image_bytes, target_size=(224,224)):
    # image_bytes: raw file bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # --- recommended preprocessing: resize, center crop or maintain aspect ratio ---
    img = img.resize(target_size, Image.BILINEAR)
    x = np.array(img).astype('float32') / 255.0   # scale to [0,1]
    # If your model used different normalization (e.g. imagenet), apply it here
    x = np.expand_dims(x, axis=0)  # shape (1,H,W,3)
    return x

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image file", 400
    f = request.files['image']
    if f.filename == '':
        return "Empty filename", 400
    if not allowed_file(f.filename):
        return "File type not allowed", 400

    filename = secure_filename(f.filename)
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(saved_path)

    # read bytes and preprocess
    with open(saved_path, 'rb') as fh:
        image_bytes = fh.read()
    x = preprocess_image(image_bytes, target_size=(224,224))  # adjust to your model input

    # model prediction
    preds = model.predict(x)  # shape (1, n_classes) or other
    # If model outputs logits or single value, adapt accordingly.
    probs = preds[0]
    # if single-output (binary), convert:
    if probs.ndim == 0:
        # binary scalar
        prob_val = float(probs)
        # choose threshold 0.5
        top_label = labels[1] if prob_val > 0.5 else labels[0]
        probs_dict = {labels[0]: 1-prob_val, labels[1]: prob_val}
    else:
        # multi-class
        # if model returned logits, apply softmax
        if not np.allclose(probs.sum(), 1.0):
            probs = tf.nn.softmax(probs).numpy()
        idx = int(np.argmax(probs))
        top_label = labels[idx] if idx < len(labels) else str(idx)
        probs_dict = {labels[i] if i < len(labels) else str(i): float(probs[i]) for i in range(len(probs))}

    response = {
        "prediction": top_label,
        "probs": probs_dict,
        "model_version": "v1.0"   # update per your versioning
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
