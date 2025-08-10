# src/evaluate_models.py
import os
import numpy as np
import json
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.models import load_model
# We need the model map to get the correct preprocessing function
from train_transfer_learning import BACKBONE_MAP 
from data_preprocessing import get_generators
from config import MODELS_DIR
from utils import copy_model

def evaluate_model_on_test(model_path, test_gen):
    """Evaluates a given model on the test data generator."""
    model = load_model(model_path)
    # Reset the generator to ensure it starts from the beginning
    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    report = classification_report(y_true, y_pred, output_dict=True, target_names=list(test_gen.class_indices.keys()))
    f1_macro = f1_score(y_true, y_pred, average='macro')
    return report, f1_macro

if __name__ == "__main__":
    # Gather model files from the models directory
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_best.h5")]
    if not model_files:
        print(f"[ERROR] No *_best.h5 files found in {MODELS_DIR} — run training first.")
        exit(1)

    best_f1 = -1.0
    best_model_file = None
    results = {}

    for model_filename in model_files:
        # Extract the backbone name from the filename (e.g., "MobileNetV2_best.h5" -> "MobileNetV2")
        backbone_name = model_filename.replace("_best.h5", "")
        
        print(f"\n[INFO] Evaluating {model_filename} ...")

        if backbone_name not in BACKBONE_MAP:
            print(f"[WARN] Backbone for {model_filename} not found in BACKBONE_MAP, skipping.")
            continue

        # Get the correct preprocessing function for this model
        _, preprocess_function = BACKBONE_MAP[backbone_name]

        # Create a new test generator with the correct preprocessing
        print(f"[INFO] Creating test generator with preprocessing for {backbone_name}...")
        _, _, test_gen = get_generators(augment=False, preprocess_func=preprocess_function)

        # Evaluate the model
        model_path = os.path.join(MODELS_DIR, model_filename)
        report, f1_macro = evaluate_model_on_test(model_path, test_gen)
        
        results[model_filename] = {'f1_macro': f1_macro, 'report': report}
        print(f"[RESULT] {model_filename} — F1_macro: {f1_macro:.4f}")

        # Check if this is the best model so far
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_file = model_path

    if best_model_file:
        print(f"\n[INFO] Best model by F1_macro: {os.path.basename(best_model_file)} (F1={best_f1:.4f})")
        # Copy the best performing model to a generic filename for the app to use
        best_dst = os.path.join(MODELS_DIR, "best_model.h5")
        copy_model(best_model_file, best_dst)
        print(f"[INFO] Copied best model to {best_dst}")

        # Save a summary of the evaluation results
        with open(os.path.join(MODELS_DIR, "evaluation_summary.json"), "w", encoding="utf-8") as f:
            # Storing only the F1 score for brevity in the summary
            summary_results = {k: {'f1_macro': v['f1_macro']} for k, v in results.items()}
            json.dump(summary_results, f, indent=2)
        print("[INFO] Saved evaluation summary.")
    else:
        print("[ERROR] Evaluation complete, but no best model was determined.")
