# 10x/Quant/setup_path.py
import os
import sys

BASE_DIR = os.path.dirname(__file__)  # this points to GenAIQuant10x/GenAIQuant

# Add Quant and its important subfolders to sys.path
sys.path.append(BASE_DIR)

# Optional: if you want to directly import from deeper paths without full package names
extra_paths = [
    os.path.join(BASE_DIR, "evaluation"),
    os.path.join(BASE_DIR, "evaluation", "metrics"),
    os.path.join(BASE_DIR, "evaluation", "metrics", "vlmevalkit"),
    os.path.join(BASE_DIR, "evaluation", "metrics", "vlmevalkit", "vlmeval"),
    os.path.join(BASE_DIR, "evaluation", "metrics", "vlmevalkit", "vlmeval", "vlm"),
    os.path.join(
        BASE_DIR, "evaluation", "metrics", "vlmevalkit", "vlmeval", "vlm", "dataset"
    ),
]

for p in extra_paths:
    if p not in sys.path:
        sys.path.append(p)
