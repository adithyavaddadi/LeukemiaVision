import os
import sys

# Add root directory to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)

# Set model path for serverless environment
os.environ["MODEL_PATH"] = os.path.join(root_dir, "models", "production.onnx")

from backend.app.main import app
