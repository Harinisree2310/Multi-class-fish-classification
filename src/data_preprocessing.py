# src/data_preprocessing.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE

def get_generators(augment=True, preprocess_func=None):
    """
    Creates and returns the training, validation, and test data generators.
    Handles both model-specific preprocessing and default rescaling.

    Args:
        augment (bool): Whether to apply data augmentation.
        preprocess_func (function, optional): A model-specific preprocessing function.
                                              If None, defaults to rescale=1./255.
    """
    # Define the base arguments for the data generator
    datagen_args = {}
    if preprocess_func:
        # Use the provided model-specific preprocessing function
        datagen_args['preprocessing_function'] = preprocess_func
    else:
        # Default to simple rescaling for custom models
        datagen_args['rescale'] = 1./255

    # Create the training data generator
    if augment:
        train_datagen = ImageDataGenerator(
            **datagen_args,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
    else:
        train_datagen = ImageDataGenerator(**datagen_args)

    # Create validation and test generators (no augmentation)
    val_test_datagen = ImageDataGenerator(**datagen_args)

    print("[INFO] Creating training data generator...")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    print("[INFO] Creating validation data generator...")
    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("[INFO] Creating test data generator...")
    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen
