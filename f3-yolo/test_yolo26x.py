import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import cv2
import numpy as np

print("Testing YOLO26x model loading...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

model_path = 'best_weights_yolo26x_dataset_t.pt'
print(f"Loading model from: {model_path}")

try:
    from ultralytics import YOLO
    print("Using ultralytics YOLO class")
    model = YOLO(model_path)
    print(f"Model type: {type(model)}")
    print(f"Model loaded successfully!")
    print(f"Model names: {model.names}")
    
    # Test inference
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(test_img)
    print(f"Inference test successful! Results: {results}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
