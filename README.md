# YOLO11n Object Detection - PyTorch and ONNX

This project demonstrates object detection using YOLO11n (the smallest YOLO11 model) with both PyTorch and ONNX inference.

## Features

- PyTorch inference with yolo11n.pt model
- Model conversion from PyTorch to ONNX format
- ONNX inference with the converted model
- Console output with detection results (bounding boxes, labels, confidence scores)
- Annotated output images for both PyTorch and ONNX inference

## Requirements

- Python 3.8+
- Virtual environment (venv)

## Installation

### 1. Clone or navigate to the project directory

```bash
cd /path/to/Yolo-test-task
```

### 2. Create a virtual environment (if not already created)

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This will install:
- ultralytics (YOLO library)
- opencv-python (image processing)
- onnx (ONNX model format)
- onnxruntime (ONNX inference)
- numpy (numerical operations)
- Pillow (image handling)
- PyTorch and related dependencies

**Note:** The installation may take several minutes as PyTorch and CUDA libraries are large packages (~2GB total).

## Usage

### Run the complete pipeline

```bash
python3 main.py
```

This will:
1. Load the yolo11n.pt model
2. Run PyTorch inference on image.jpeg
3. Convert the model to ONNX format (yolo11n.onnx)
4. Run ONNX inference on the same image

### Output Files

After running the script, you'll get:
- `output_pytorch.jpeg` - Annotated image from PyTorch inference
- `yolo11n.onnx` - Converted ONNX model
- `output_onnx.jpeg` - Annotated image from ONNX inference
- Console output with detailed detection results

## Project Structure

```
.
├── main.py              # Main script with all inference logic
├── requirements.txt     # Python dependencies
├── yolo11n.pt          # YOLO11n PyTorch model
├── image.jpeg          # Input image for detection
└── README.md           # This file
```

## Example Output

The script will print detection results to the console:

```
============================================================
PYTORCH INFERENCE
============================================================
✓ Loaded PyTorch model: yolo11n.pt
✓ Detected 3 objects in image.jpeg

Detection Results:
------------------------------------------------------------
Object 1:
  Class: person
  Confidence: 0.8945
  Bounding Box: (120.45, 85.32, 456.78, 512.90)
...
```

## Troubleshooting

### ModuleNotFoundError
Make sure you've activated the virtual environment and installed all dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### CUDA/GPU Issues
The script will automatically use CPU if CUDA is not available. For GPU acceleration, ensure you have compatible NVIDIA drivers installed.

## Dependencies

See `requirements.txt` for the complete list of dependencies.