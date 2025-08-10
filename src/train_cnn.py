# src/train_cnn.py
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_preprocessing import get_generators
from config import MODELS_DIR, IMG_SIZE, EPOCHS, LEARNING_RATE

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    train_gen, val_gen, _ = get_generators(augment=True)
    num_classes = train_gen.num_classes
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

    model = build_cnn(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    os.makedirs(MODELS_DIR, exist_ok=True)
    ckp_path = os.path.join(MODELS_DIR, "cnn_best.h5")
    checkpoint = ModelCheckpoint(ckp_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint, earlystop])
    print(f"[INFO] CNN training finished. Best model (if saved) at {ckp_path}")
