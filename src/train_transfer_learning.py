# src/train_transfer_learning.py
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# We need to import the preprocessing functions for each model
from tensorflow.keras.applications import vgg16, resnet50, mobilenet_v2, inception_v3, efficientnet
from data_preprocessing import get_generators
from config import MODELS_DIR, IMG_SIZE, EPOCHS, LEARNING_RATE, BACKBONES
from utils import save_class_names

# Map the model name not just to the model class, but also to its preprocess_input function
BACKBONE_MAP = {
    "VGG16": (VGG16, vgg16.preprocess_input),
    "ResNet50": (ResNet50, resnet50.preprocess_input),
    "MobileNetV2": (MobileNetV2, mobilenet_v2.preprocess_input),
    "InceptionV3": (InceptionV3, inception_v3.preprocess_input),
    "EfficientNetB0": (EfficientNetB0, efficientnet.preprocess_input)
}

def build_head(base_model, num_classes):
    """Builds the classification head on top of the base model."""
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

if __name__ == "__main__":
    # Loop through each backbone defined in the config
    for backbone_name in BACKBONES:
        print(f"\n===== PREPARING TO TRAIN: {backbone_name} =====")
        
        if backbone_name not in BACKBONE_MAP:
            print(f"[WARN] Backbone {backbone_name} not found in BACKBONE_MAP, skipping.")
            continue

        # Get the model class and its specific preprocessing function
        Base, preprocess_function = BACKBONE_MAP[backbone_name]

        # Get data generators with the CORRECT preprocessing function for the current model
        train_gen, val_gen, _ = get_generators(augment=True, preprocess_func=preprocess_function)
        
        # Check if this is the first run to save class names
        if 'class_names' not in locals():
            num_classes = train_gen.num_classes
            class_names = list(train_gen.class_indices.keys())
            print("[INFO] Class names (training order):", class_names)
            # Save class names file so prediction & app can use it
            save_class_names(class_names, os.path.join(MODELS_DIR, "class_names.txt"))

        # Build the model
        base_model = Base(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        # Freeze the base model layers
        base_model.trainable = False

        model = build_head(base_model, num_classes)
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Set up callbacks
        os.makedirs(MODELS_DIR, exist_ok=True)
        # CORRECTED THE TYPO ON THE LINE BELOW
        checkpoint_path = os.path.join(MODELS_DIR, f"{backbone_name}_best.h5")
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
        earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        # Train the model
        print(f"[INFO] Starting training for {backbone_name}...")
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint, earlystop])
        print(f"[INFO] Finished training {backbone_name}. Best model saved to {checkpoint_path}")
