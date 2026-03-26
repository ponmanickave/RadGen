import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# =====================================================
# PATHS (Relative to project root)
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "multimodal_report_generator.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")
MAX_SEQ_PATH = os.path.join(BASE_DIR, "max_sequence_length.txt")

class AIHandler:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.cnn = None
        self.max_seq_len = None
        self.index_word = None

    def load_resources(self):
        """Lazy loading of heavy models"""
        if self.tokenizer is None:
            print("🔹 Loading tokenizer...")
            with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
                self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
            self.index_word = {v: k for k, v in self.tokenizer.word_index.items()}

        if self.max_seq_len is None:
            with open(MAX_SEQ_PATH, "r") as f:
                self.max_seq_len = int(f.read().strip())

        if self.model is None:
            print("🔹 Loading trained multimodal model...")
            self.model = load_model(MODEL_PATH)

        if self.cnn is None:
            print("🔹 Loading ResNet50...")
            self.cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    def clean_section(self, text, section):
        text = text.lower()
        if section == "lungs":
            if any(k in text for k in ["opacity", "consolidation", "infiltrate"]):
                return "Bilateral pulmonary opacities are noted, suggestive of parenchymal involvement."
            return "The lungs are clear bilaterally with no focal consolidation or abnormal opacities."

        if section == "heart":
            if any(k in text for k in ["enlarged", "cardiomegaly"]):
                return "The cardiac silhouette appears enlarged, consistent with cardiomegaly."
            return "The cardiac silhouette is normal in size and configuration."

        if section == "pleura":
            if any(k in text for k in ["effusion", "pneumothorax"]):
                return "There is evidence of pleural effusion or pneumothorax."
            return "No pleural effusion or pneumothorax is identified."

        if section == "bones":
            if any(k in text for k in ["fracture", "lesion", "degenerative"]):
                return "Osseous abnormalities are noted within the visualized skeletal structures."
            return "The visualized bony thorax demonstrates no acute osseous abnormality."

        return "No abnormality detected."

    def extract_image_features(self, img_path):
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return self.cnn.predict(img)

    def generate_report_content(self, image_path):
        self.load_resources()
        
        print("\n🩺 Extracting image features...")
        image_features = self.extract_image_features(image_path)

        dummy_seq = np.zeros((1, self.max_seq_len))
        preds = self.model.predict([image_features, dummy_seq])

        # TOP-K DECODING
        top_k = 25
        top_indices = preds[0].argsort()[-top_k:][::-1]
        pred_words = [self.index_word.get(int(i), "") for i in top_indices]

        raw_text = " ".join(pred_words)

        report_data = {
            "exam": "Chest X-ray (PA View)",
            "history": "Routine clinical evaluation.",
            "findings": {
                "lungs": self.clean_section(raw_text, "lungs"),
                "heart": self.clean_section(raw_text, "heart"),
                "pleura": self.clean_section(raw_text, "pleura"),
                "bones": self.clean_section(raw_text, "bones"),
            },
            "impression": "No acute cardiopulmonary abnormality detected.",
            "recommendation": "Clinical correlation advised."
        }
        
        return report_data
