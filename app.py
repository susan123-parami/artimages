from flask import Flask, render_template, request
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model and metadata
model = tf.keras.models.load_model('model/classifier.keras')

with open('model/metrics.pkl', 'rb') as f:
    metadata = pickle.load(f)

CLASS_NAMES = metadata['class_names']
IMG_SIZE = tuple(metadata['img_size'])
BEST_MODEL_NAME = metadata.get('best_model', 'Model')

# Below this top-probability, the prediction is shown with a "low confidence" warning
LOW_CONFIDENCE_THRESHOLD = 0.55

# Folder for user uploads
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


# Quiz images: pre-compute the model's predictions at startup so the
# "Can you beat the model?" section uses the real saved CNN, not a stub.
QUIZ_SOURCES = [
    ('quiz1.jpg', 'painting'),
    ('quiz2.jpg', 'iconography'),
    ('quiz3.jpg', 'engraving'),
]
QUIZ_DATA = []
for filename, truth in QUIZ_SOURCES:
    fpath = os.path.join('static/img', filename)
    if not os.path.isfile(fpath):
        continue
    img = load_img(fpath, target_size=IMG_SIZE)
    arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    QUIZ_DATA.append({
        'image': filename,
        'truth': truth,
        'model_pick': CLASS_NAMES[idx],
        'model_conf': round(float(preds[idx]) * 100, 1),
    })


@app.route('/')
def index():
    return render_template('index.html', quiz=QUIZ_DATA)


@app.route('/predict', methods=['POST'])
def predict():
    # Check file in request
    if 'image' not in request.files:
        return render_template('index.html', error='No file uploaded', quiz=QUIZ_DATA)

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No file selected', quiz=QUIZ_DATA)

    if not allowed_file(file.filename):
        return render_template('index.html', error='Only PNG, JPG, JPEG files allowed', quiz=QUIZ_DATA)

    # Save uploaded image
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess image (same as training: resize + rescale)
    img = load_img(filepath, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(preds)
    prediction = CLASS_NAMES[pred_idx]
    top_confidence = float(preds[pred_idx])
    low_confidence = top_confidence < LOW_CONFIDENCE_THRESHOLD

    # Build probability dict
    prob_dict = {}
    for i in range(len(CLASS_NAMES)):
        prob_dict[CLASS_NAMES[i]] = round(float(preds[i]) * 100, 1)

    return render_template('result.html',
                           page='predict',
                           prediction=prediction,
                           probabilities=prob_dict,
                           uploaded_image=filename,
                           confidence=round(top_confidence * 100, 1),
                           low_confidence=low_confidence,
                           model_name=BEST_MODEL_NAME)


@app.route('/performance')
def performance():
    return render_template('result.html',
                           page='performance',
                           metrics=metadata['models_metrics'],
                           best_model=metadata['best_model'])


@app.route('/explore')
def explore():
    return render_template('result.html', page='explore')


if __name__ == '__main__':
    app.run(debug=True)
