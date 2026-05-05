import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from src.preprocess.forensics import forensic_score

# load models
cnn = tf.keras.models.load_model("src/models/cnn.keras")
exif = joblib.load("src/models/exif.pkl")


# =========================
# CNN PREDICTION
# =========================
def img_pred(path):
    img = image.load_img(path, target_size=(224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, 0)

    return float(cnn.predict(arr)[0][0])


# =========================
# EXIF PREDICTION
# =========================
def exif_pred(features):
    arr = np.array([[features["missing"],
                     features["has_exif"],
                     features["suspicious"]]])

    return exif.predict_proba(arr)[0][1]


# =========================
# FINAL FUSION MODEL (UPDATED)
# =========================
def final_predict(path, features):

    cnn_score = img_pred(path)
    exif_score = exif_pred(features)
    forensic = forensic_score(path)

    # 🔥 NEW MULTI-SIGNAL FUSION
    final_score = (
        0.5 * cnn_score +
        0.2 * exif_score +
        0.3 * forensic
    )

    return {
        "cnn_score": cnn_score,
        "exif_score": exif_score,
        "forensic_score": forensic,
        "final_score": final_score,
        "result": "REAL" if final_score > 0.5 else "FAKE"
    }