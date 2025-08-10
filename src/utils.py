# src/utils.py
import os
import json
import shutil

def save_class_names(class_names, path):
    with open(path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"[INFO] Saved {len(class_names)} class names to {path}")

def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def copy_model(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
    print(f"[INFO] Copied {src} -> {dst}")
