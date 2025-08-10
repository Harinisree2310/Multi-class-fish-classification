# src/extract_class_names.py
import os
from config import TRAIN_DIR, CLASS_NAMES_FILE
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Train folder not found at {TRAIN_DIR}. Please check path.")

# Use Keras flow_from_directory to guarantee same order as training
datagen = ImageDataGenerator(rescale=1./255)
gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224,224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
class_names = list(gen.class_indices.keys())
print("Detected classes:", class_names)

# Save to file
with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
    for c in class_names:
        f.write(c + "\n")
print(f"[INFO] Wrote class names to {CLASS_NAMES_FILE}")
