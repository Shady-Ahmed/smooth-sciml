#!/usr/bin/env python3
"""
Clean script to remove all generated datasets, checkpoints, and figures
from the Chapter 7 companion directory.
"""
import glob
import os
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_PATTERNS = [
    # Artifact folders
    "**/outputs",
    "**/plots",
    "**/burgers-data",
    "**/models",
    "**/data",
    # Python cache
    "**/__pycache__",
    "**/.pytest_cache",
]

TARGET_FILES = [
    "**/*.npy",
    "**/*.npz",
    "**/*.eqx",
    "**/*.pkl",
    "**/*.pickle",
    "**/*.png",
    "**/*.pdf",
    "**/.DS_Store",
]

def clean():
    count_dirs = 0
    count_files = 0

    # Remove files
    for pat in TARGET_FILES:
        for f in glob.glob(os.path.join(ROOT_DIR, pat), recursive=True):
            if os.path.isfile(f) and not f.endswith("main.tex"):
                try:
                    os.remove(f)
                    count_files += 1
                except Exception as e:
                    print(f"Error removing {f}: {e}")

    # Remove directories
    for pat in TARGET_PATTERNS:
        for d in glob.glob(os.path.join(ROOT_DIR, pat), recursive=True):
            if os.path.isdir(d):
                try:
                    shutil.rmtree(d)
                    count_dirs += 1
                except Exception as e:
                    print(f"Error removing {d}: {e}")

    print(f"Clean complete: removed {count_files} files and {count_dirs} directories.")

if __name__ == "__main__":
    clean()
